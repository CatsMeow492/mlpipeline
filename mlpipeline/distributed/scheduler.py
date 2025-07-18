"""
Distributed task scheduler for coordinating work across multiple backends.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from queue import Queue, PriorityQueue
import json

from ..core.errors import MLPipelineError
from .resource_manager import ResourceManager, ResourceRequirement

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class Task:
    """Distributed task definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    resource_requirements: Optional[ResourceRequirement] = None
    backend_preference: Optional[str] = None
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare tasks by priority for priority queue."""
        return self.priority.value < other.priority.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.name,
            "status": self.status.name,
            "resource_requirements": self.resource_requirements.to_dict() if self.resource_requirements else None,
            "backend_preference": self.backend_preference,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": self.metadata
        }


class DistributedScheduler:
    """Scheduler for distributing tasks across multiple computing backends."""
    
    def __init__(self, 
                 resource_manager: ResourceManager,
                 max_concurrent_tasks: int = 10,
                 task_timeout: int = 3600,
                 heartbeat_interval: int = 30):
        """
        Initialize distributed scheduler.
        
        Args:
            resource_manager: Resource manager instance
            max_concurrent_tasks: Maximum concurrent tasks
            task_timeout: Default task timeout in seconds
            heartbeat_interval: Heartbeat interval for monitoring
        """
        self.resource_manager = resource_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Task queues and tracking
        self.task_queue = PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_futures: Dict[str, Future] = {}
        
        # Thread pool for task execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Scheduler state
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_execution_time": 0.0
        }
        
        logger.info("Distributed scheduler initialized")
    
    def start(self) -> None:
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        logger.info("Distributed scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all running tasks
        for task_id in list(self.running_tasks.keys()):
            self.cancel_task(task_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Distributed scheduler stopped")
    
    def submit_task(self, 
                   func: Callable,
                   *args,
                   name: str = "",
                   priority: TaskPriority = TaskPriority.NORMAL,
                   resource_requirements: Optional[ResourceRequirement] = None,
                   backend_preference: Optional[str] = None,
                   timeout: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            name: Task name
            priority: Task priority
            resource_requirements: Resource requirements
            backend_preference: Preferred backend
            timeout: Task timeout
            metadata: Additional metadata
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            resource_requirements=resource_requirements,
            backend_preference=backend_preference,
            timeout=timeout or self.task_timeout,
            metadata=metadata or {}
        )
        
        self.task_queue.put(task)
        self.stats["tasks_submitted"] += 1
        
        logger.info(f"Task submitted: {task.id} ({task.name})")
        return task.id
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        
        # Check queue
        with self.task_queue.mutex:
            for task in self.task_queue.queue:
                if task.id == task_id:
                    return task.status
        
        return None
    
    def get_task_result(self, task_id: str) -> Any:
        """Get result of a completed task."""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise MLPipelineError(f"Task failed: {task.error}")
        
        raise MLPipelineError(f"Task not found or not completed: {task_id}")
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        # Cancel running task
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # Cancel future if exists
            if task_id in self.task_futures:
                future = self.task_futures[task_id]
                future.cancel()
                del self.task_futures[task_id]
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.running_tasks[task_id]
            
            self.stats["tasks_cancelled"] += 1
            logger.info(f"Task cancelled: {task_id}")
            return True
        
        # Cancel queued task
        with self.task_queue.mutex:
            queue_items = list(self.task_queue.queue)
            self.task_queue.queue.clear()
            
            cancelled = False
            for task in queue_items:
                if task.id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.completed_tasks[task_id] = task
                    self.stats["tasks_cancelled"] += 1
                    cancelled = True
                    logger.info(f"Task cancelled: {task_id}")
                else:
                    self.task_queue.put(task)
            
            return cancelled
        
        return False
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List tasks with optional status filter."""
        tasks = []
        
        # Add running tasks
        for task in self.running_tasks.values():
            if status is None or task.status == status:
                tasks.append(task.to_dict())
        
        # Add completed tasks
        for task in self.completed_tasks.values():
            if status is None or task.status == status:
                tasks.append(task.to_dict())
        
        # Add queued tasks
        with self.task_queue.mutex:
            for task in self.task_queue.queue:
                if status is None or task.status == status:
                    tasks.append(task.to_dict())
        
        return sorted(tasks, key=lambda x: x["created_at"], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        current_stats = self.stats.copy()
        current_stats.update({
            "queued_tasks": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "average_execution_time": (
                self.stats["total_execution_time"] / max(1, self.stats["tasks_completed"])
            ),
            "success_rate": (
                self.stats["tasks_completed"] / max(1, self.stats["tasks_submitted"]) * 100
            )
        })
        return current_stats
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.running:
            try:
                # Check if we can run more tasks
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    time.sleep(1)
                    continue
                
                # Get next task from queue
                try:
                    task = self.task_queue.get(timeout=1)
                except:
                    continue
                
                # Check resource availability
                if task.resource_requirements:
                    if not self.resource_manager.check_resource_availability(task.resource_requirements):
                        # Put task back in queue and wait
                        self.task_queue.put(task)
                        time.sleep(5)
                        continue
                
                # Execute task
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: Task) -> None:
        """Execute a task."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.running_tasks[task.id] = task
        
        logger.info(f"Executing task: {task.id} ({task.name})")
        
        # Submit to thread pool
        future = self.executor.submit(self._run_task, task)
        self.task_futures[task.id] = future
        
        # Add callback for completion
        future.add_done_callback(lambda f: self._task_completed(task.id, f))
    
    def _run_task(self, task: Task) -> Any:
        """Run a task function."""
        try:
            # Select backend if preference specified
            backend = None
            if task.backend_preference:
                backend = self.resource_manager.backends.get(task.backend_preference)
                if backend and hasattr(backend, 'submit_task'):
                    # Use backend's task submission
                    return backend.submit_task(task.func, *task.args, **task.kwargs)
            
            # Execute function directly
            return task.func(*task.args, **task.kwargs)
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.id} - {e}")
            raise
    
    def _task_completed(self, task_id: str, future: Future) -> None:
        """Handle task completion."""
        if task_id not in self.running_tasks:
            return
        
        task = self.running_tasks[task_id]
        task.completed_at = time.time()
        
        # Calculate execution time
        execution_time = task.completed_at - task.started_at
        self.stats["total_execution_time"] += execution_time
        
        try:
            # Get result
            task.result = future.result()
            task.status = TaskStatus.COMPLETED
            self.stats["tasks_completed"] += 1
            logger.info(f"Task completed: {task_id} ({execution_time:.2f}s)")
            
        except Exception as e:
            # Handle failure
            task.error = str(e)
            task.status = TaskStatus.FAILED
            self.stats["tasks_failed"] += 1
            logger.error(f"Task failed: {task_id} - {e}")
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.error = None
                
                # Put back in queue
                self.task_queue.put(task)
                logger.info(f"Retrying task: {task_id} (attempt {task.retry_count + 1})")
                
                # Remove from running tasks
                del self.running_tasks[task_id]
                if task_id in self.task_futures:
                    del self.task_futures[task_id]
                return
        
        # Move to completed tasks
        self.completed_tasks[task_id] = task
        del self.running_tasks[task_id]
        if task_id in self.task_futures:
            del self.task_futures[task_id]
    
    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for monitoring."""
        while self.running:
            try:
                # Monitor resource usage
                self.resource_manager.monitor_resources()
                
                # Check for timed out tasks
                current_time = time.time()
                timed_out_tasks = []
                
                for task in self.running_tasks.values():
                    if task.timeout and (current_time - task.started_at) > task.timeout:
                        timed_out_tasks.append(task.id)
                
                # Cancel timed out tasks
                for task_id in timed_out_tasks:
                    logger.warning(f"Task timed out: {task_id}")
                    self.cancel_task(task_id)
                
                # Clean up old completed tasks (keep last 1000)
                if len(self.completed_tasks) > 1000:
                    sorted_tasks = sorted(
                        self.completed_tasks.items(),
                        key=lambda x: x[1].completed_at or 0
                    )
                    # Keep last 1000 tasks
                    tasks_to_keep = dict(sorted_tasks[-1000:])
                    self.completed_tasks = tasks_to_keep
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.heartbeat_interval)
    
    def wait_for_task(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """Wait for a task to complete and return its result."""
        start_time = time.time()
        
        while True:
            status = self.get_task_status(task_id)
            
            if status == TaskStatus.COMPLETED:
                return self.get_task_result(task_id)
            elif status == TaskStatus.FAILED:
                task = self.completed_tasks.get(task_id)
                if task:
                    raise MLPipelineError(f"Task failed: {task.error}")
                else:
                    raise MLPipelineError(f"Task failed: {task_id}")
            elif status == TaskStatus.CANCELLED:
                raise MLPipelineError(f"Task was cancelled: {task_id}")
            elif status is None:
                raise MLPipelineError(f"Task not found: {task_id}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise MLPipelineError(f"Timeout waiting for task: {task_id}")
            
            time.sleep(1)
    
    def wait_for_all(self, task_ids: List[str], timeout: Optional[int] = None) -> List[Any]:
        """Wait for multiple tasks to complete."""
        results = []
        for task_id in task_ids:
            result = self.wait_for_task(task_id, timeout)
            results.append(result)
        return results


def create_distributed_scheduler(resource_manager: ResourceManager, 
                               config: Dict[str, Any]) -> DistributedScheduler:
    """Factory function to create distributed scheduler."""
    return DistributedScheduler(
        resource_manager=resource_manager,
        max_concurrent_tasks=config.get('max_concurrent_tasks', 10),
        task_timeout=config.get('task_timeout', 3600),
        heartbeat_interval=config.get('heartbeat_interval', 30)
    )