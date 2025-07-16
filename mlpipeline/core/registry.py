"""Component registry for dynamic component loading and management."""

from typing import Dict, Type, Optional
import logging
from .interfaces import PipelineComponent, ComponentType


class ComponentRegistry:
    """Registry for managing pipeline components."""
    
    def __init__(self):
        self._components: Dict[str, Type[PipelineComponent]] = {}
        self._instances: Dict[str, PipelineComponent] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_component(self, name: str, component_class: Type[PipelineComponent]) -> None:
        """Register a component class with a given name."""
        if not issubclass(component_class, PipelineComponent):
            raise ValueError(f"Component {component_class} must inherit from PipelineComponent")
        
        self._components[name] = component_class
        self.logger.info(f"Registered component: {name}")
    
    def get_component_class(self, name: str) -> Optional[Type[PipelineComponent]]:
        """Get a component class by name."""
        return self._components.get(name)
    
    def create_component(self, name: str, **kwargs) -> Optional[PipelineComponent]:
        """Create an instance of a registered component."""
        component_class = self.get_component_class(name)
        if component_class is None:
            self.logger.error(f"Component {name} not found in registry")
            return None
        
        try:
            instance = component_class(**kwargs)
            self._instances[name] = instance
            self.logger.info(f"Created component instance: {name}")
            return instance
        except Exception as e:
            self.logger.error(f"Failed to create component {name}: {str(e)}")
            return None
    
    def get_component_instance(self, name: str) -> Optional[PipelineComponent]:
        """Get an existing component instance."""
        return self._instances.get(name)
    
    def list_components(self) -> Dict[str, Type[PipelineComponent]]:
        """List all registered components."""
        return self._components.copy()
    
    def unregister_component(self, name: str) -> bool:
        """Unregister a component."""
        if name in self._components:
            del self._components[name]
            if name in self._instances:
                del self._instances[name]
            self.logger.info(f"Unregistered component: {name}")
            return True
        return False


# Global component registry instance
component_registry = ComponentRegistry()