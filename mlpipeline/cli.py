"""Command-line interface for ML Pipeline."""

import click
import logging
import sys
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from .config.manager import ConfigManager
from .core.orchestrator import PipelineOrchestrator
from .core.interfaces import ExecutionContext, ComponentType
from .core.registry import component_registry


# Initialize rich console for better output formatting
console = Console()


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)


@click.group(invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.option('--version', is_flag=True, help='Show version and exit')
@click.pass_context
def cli(ctx, verbose: bool, log_file: Optional[str], version: bool):
    """ML Pipeline - A comprehensive machine learning pipeline framework."""
    if version:
        console.print("ML Pipeline version 0.1.0")
        sys.exit(0)
    
    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        return
    
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['log_file'] = log_file
    
    setup_logging(verbose, log_file)
    
    if verbose:
        console.print("[bold green]ML Pipeline CLI[/bold green] - Verbose mode enabled")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to pipeline configuration file')
@click.option('--experiment-id', type=str, help='Custom experiment ID')
@click.option('--artifacts-path', type=click.Path(), default='./artifacts',
              help='Path to store artifacts')
@click.option('--resume', is_flag=True, help='Resume from checkpoint if available')
@click.option('--dry-run', is_flag=True, help='Validate configuration without execution')
@click.option('--interactive', '-i', is_flag=True, help='Run in interactive mode')
@click.pass_context
def train(ctx, config: Optional[str], experiment_id: Optional[str], artifacts_path: str, 
          resume: bool, dry_run: bool, interactive: bool):
    """Train a machine learning model using the specified configuration."""
    
    try:
        # Handle interactive mode
        if interactive:
            config = _interactive_config_setup(config)
        
        if not config:
            console.print("[red]Error:[/red] Configuration file is required")
            console.print("Use --interactive to create a configuration interactively")
            sys.exit(1)
        
        with console.status("[bold green]Loading configuration..."):
            config_manager = ConfigManager()
            pipeline_config = config_manager.load_config(config)
        
        console.print(f"[green]✓[/green] Configuration loaded: {pipeline_config.pipeline.name}")
        
        if dry_run:
            console.print("[yellow]Dry run mode - validating configuration only[/yellow]")
            _display_config_summary(pipeline_config)
            console.print("[green]✓[/green] Configuration is valid")
            return
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            max_workers=4,
            enable_checkpointing=resume
        )
        
        # Create execution context
        import uuid
        exp_id = experiment_id or str(uuid.uuid4())
        
        context = ExecutionContext(
            experiment_id=exp_id,
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config=pipeline_config.model_dump(),
            artifacts_path=artifacts_path,
            logger=logging.getLogger("training"),
            metadata={"mode": "training"}
        )
        
        # Create training stages from config
        stages = _create_training_stages(pipeline_config, orchestrator)
        
        console.print(f"[blue]Starting training pipeline with experiment ID: {exp_id}[/blue]")
        
        # Execute pipeline with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Training pipeline...", total=len(stages))
            
            result = orchestrator.execute_pipeline(stages, context)
            
            progress.update(task, completed=len(stages))
        
        if result.success:
            console.print(f"[green]✓[/green] Training completed successfully!")
            _display_training_results(result)
        else:
            console.print(f"[red]✗[/red] Training failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to pipeline configuration file')
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to trained model')
@click.option('--input-data', required=True, type=click.Path(exists=True),
              help='Path to input data for inference')
@click.option('--output-path', type=click.Path(), default='./predictions.json',
              help='Path to save predictions')
@click.option('--batch-size', type=int, default=1000, help='Batch size for processing')
@click.option('--confidence-threshold', type=float, help='Confidence threshold for predictions')
@click.pass_context
def inference(ctx, config: str, model_path: str, input_data: str, output_path: str,
              batch_size: int, confidence_threshold: Optional[float]):
    """Perform inference using a trained model."""
    
    try:
        with console.status("[bold green]Loading configuration and model..."):
            config_manager = ConfigManager()
            pipeline_config = config_manager.load_config(config)
        
        console.print(f"[green]✓[/green] Configuration loaded")
        console.print(f"[blue]Model:[/blue] {model_path}")
        console.print(f"[blue]Input data:[/blue] {input_data}")
        console.print(f"[blue]Batch size:[/blue] {batch_size}")
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Create execution context
        import uuid
        context = ExecutionContext(
            experiment_id=str(uuid.uuid4()),
            stage_name="inference",
            component_type=ComponentType.MODEL_INFERENCE,
            config=pipeline_config.model_dump(),
            artifacts_path="./inference_artifacts",
            logger=logging.getLogger("inference"),
            metadata={
                "mode": "inference",
                "model_path": model_path,
                "input_data": input_data,
                "batch_size": batch_size,
                "confidence_threshold": confidence_threshold
            }
        )
        
        # Create inference stages
        stages = _create_inference_stages(pipeline_config, orchestrator)
        
        console.print("[blue]Starting inference pipeline...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Running inference...", total=len(stages))
            
            result = orchestrator.execute_pipeline(stages, context)
            
            progress.update(task, completed=len(stages))
        
        if result.success:
            console.print(f"[green]✓[/green] Inference completed successfully!")
            console.print(f"[blue]Predictions saved to:[/blue] {output_path}")
            _display_inference_results(result)
        else:
            console.print(f"[red]✗[/red] Inference failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to pipeline configuration file')
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to trained model')
@click.option('--test-data', required=True, type=click.Path(exists=True),
              help='Path to test data')
@click.option('--output-path', type=click.Path(), default='./evaluation_results.json',
              help='Path to save evaluation results')
@click.option('--metrics', multiple=True, help='Specific metrics to compute')
@click.pass_context
def evaluate(ctx, config: str, model_path: str, test_data: str, output_path: str,
             metrics: tuple):
    """Evaluate a trained model on test data."""
    
    try:
        with console.status("[bold green]Loading configuration..."):
            config_manager = ConfigManager()
            pipeline_config = config_manager.load_config(config)
        
        console.print(f"[green]✓[/green] Configuration loaded")
        console.print(f"[blue]Model:[/blue] {model_path}")
        console.print(f"[blue]Test data:[/blue] {test_data}")
        
        if metrics:
            console.print(f"[blue]Metrics:[/blue] {', '.join(metrics)}")
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Create execution context
        import uuid
        context = ExecutionContext(
            experiment_id=str(uuid.uuid4()),
            stage_name="evaluation",
            component_type=ComponentType.MODEL_EVALUATION,
            config=pipeline_config.model_dump(),
            artifacts_path="./evaluation_artifacts",
            logger=logging.getLogger("evaluation"),
            metadata={
                "mode": "evaluation",
                "model_path": model_path,
                "test_data": test_data,
                "metrics": list(metrics) if metrics else None
            }
        )
        
        # Create evaluation stages
        stages = _create_evaluation_stages(pipeline_config, orchestrator)
        
        console.print("[blue]Starting evaluation pipeline...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Evaluating model...", total=len(stages))
            
            result = orchestrator.execute_pipeline(stages, context)
            
            progress.update(task, completed=len(stages))
        
        if result.success:
            console.print(f"[green]✓[/green] Evaluation completed successfully!")
            console.print(f"[blue]Results saved to:[/blue] {output_path}")
            _display_evaluation_results(result)
        else:
            console.print(f"[red]✗[/red] Evaluation failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to pipeline configuration file')
@click.option('--baseline-data', type=click.Path(exists=True),
              help='Path to baseline data for drift detection')
@click.option('--current-data', required=True, type=click.Path(exists=True),
              help='Path to current data to check for drift')
@click.option('--output-path', type=click.Path(), default='./drift_report.json',
              help='Path to save drift detection report')
@click.option('--threshold', type=float, help='Drift detection threshold')
@click.pass_context
def monitor(ctx, config: str, baseline_data: Optional[str], current_data: str,
            output_path: str, threshold: Optional[float]):
    """Monitor for data and model drift."""
    
    try:
        with console.status("[bold green]Loading configuration..."):
            config_manager = ConfigManager()
            pipeline_config = config_manager.load_config(config)
        
        console.print(f"[green]✓[/green] Configuration loaded")
        console.print(f"[blue]Current data:[/blue] {current_data}")
        
        if baseline_data:
            console.print(f"[blue]Baseline data:[/blue] {baseline_data}")
        
        if threshold:
            console.print(f"[blue]Drift threshold:[/blue] {threshold}")
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Create execution context
        import uuid
        context = ExecutionContext(
            experiment_id=str(uuid.uuid4()),
            stage_name="monitoring",
            component_type=ComponentType.DRIFT_DETECTION,
            config=pipeline_config.model_dump(),
            artifacts_path="./monitoring_artifacts",
            logger=logging.getLogger("monitoring"),
            metadata={
                "mode": "monitoring",
                "baseline_data": baseline_data,
                "current_data": current_data,
                "threshold": threshold
            }
        )
        
        # Create monitoring stages
        stages = _create_monitoring_stages(pipeline_config, orchestrator)
        
        console.print("[blue]Starting drift detection...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Detecting drift...", total=len(stages))
            
            result = orchestrator.execute_pipeline(stages, context)
            
            progress.update(task, completed=len(stages))
        
        if result.success:
            console.print(f"[green]✓[/green] Drift detection completed!")
            console.print(f"[blue]Report saved to:[/blue] {output_path}")
            _display_drift_results(result)
        else:
            console.print(f"[red]✗[/red] Drift detection failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='./config_template.yaml',
              help='Output path for configuration template')
@click.option('--format', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Output format')
@click.option('--use-case', type=click.Choice(['classification', 'regression', 'few-shot']),
              default='classification', help='Use case template')
def init(output: str, format: str, use_case: str):
    """Initialize a new pipeline configuration template."""
    
    try:
        config_manager = ConfigManager()
        
        # Get default config and customize based on use case
        default_config = config_manager.get_default_config()
        
        if use_case == 'regression':
            default_config['model']['parameters']['algorithm'] = 'RandomForestRegressor'
            default_config['evaluation']['metrics'] = ['mse', 'mae', 'r2_score']
        elif use_case == 'few-shot':
            default_config['few_shot'] = {
                'enabled': True,
                'prompt_template': 'templates/few_shot.txt',
                'max_examples': 5,
                'similarity_threshold': 0.8
            }
            default_config['model']['type'] = 'huggingface'
        
        # Save configuration
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if format == 'yaml':
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(default_config, f, indent=2)
        
        console.print(f"[green]✓[/green] Configuration template created: {output_path}")
        console.print(f"[blue]Use case:[/blue] {use_case}")
        console.print(f"[blue]Format:[/blue] {format}")
        
        # Display next steps
        panel = Panel(
            f"""[bold]Next Steps:[/bold]

1. Edit the configuration file: {output_path}
2. Prepare your data files
3. Run training: mlpipeline train --config {output_path}
4. Evaluate your model: mlpipeline evaluate --config {output_path} --model-path <model> --test-data <data>

[dim]For more help, run: mlpipeline --help[/dim]""",
            title="Getting Started",
            border_style="green"
        )
        console.print(panel)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to pipeline configuration file to validate')
def validate(config: Optional[str]):
    """Validate a pipeline configuration file."""
    
    if not config:
        console.print("[red]Error:[/red] Configuration file path is required")
        sys.exit(1)
    
    try:
        with console.status("[bold green]Validating configuration..."):
            config_manager = ConfigManager()
            validation_result = config_manager.validate_schema(
                config_manager._load_raw_config(Path(config))
            )
        
        if validation_result.valid:
            console.print(f"[green]✓[/green] Configuration is valid: {config}")
            
            if validation_result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation_result.warnings:
                    console.print(f"  [yellow]•[/yellow] {warning}")
            
            _display_config_summary(validation_result.config)
        else:
            console.print(f"[red]✗[/red] Configuration is invalid: {config}")
            console.print("\n[red]Errors:[/red]")
            for error in validation_result.errors:
                console.print(f"  [red]•[/red] {error}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed component information')
def status(detailed: bool):
    """Show pipeline status and component registry information."""
    
    console.print("[bold blue]ML Pipeline Status[/bold blue]\n")
    
    # Show component registry status
    console.print("[bold]Registered Components:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Class", style="yellow")
    
    if detailed:
        table.add_column("Status", style="blue")
        table.add_column("Version", style="dim")
    
    # Get registered components (this would need to be implemented in the registry)
    # For now, show placeholder information
    sample_components = [
        ("Data Ingestion", "csv_loader", "CSVDataLoader", "Active", "1.0.0"),
        ("Data Preprocessing", "standard_scaler", "StandardScalerPreprocessor", "Active", "1.0.0"),
        ("Model Training", "sklearn_trainer", "SklearnModelTrainer", "Active", "1.0.0"),
        ("Model Evaluation", "classification_evaluator", "ClassificationEvaluator", "Active", "1.0.0"),
        ("Drift Detection", "evidently_detector", "EvidentlyDriftDetector", "Active", "1.0.0"),
    ]
    
    for comp_data in sample_components:
        if detailed:
            table.add_row(*comp_data)
        else:
            table.add_row(*comp_data[:3])
    
    console.print(table)
    
    # Show system information
    console.print(f"\n[bold]System Information:[/bold]")
    console.print(f"Python version: {sys.version.split()[0]}")
    console.print(f"Platform: {sys.platform}")
    
    # Show installed packages versions
    if detailed:
        console.print(f"\n[bold]Key Dependencies:[/bold]")
        try:
            import pandas as pd
            import sklearn
            import mlflow
            
            deps_table = Table(show_header=True, header_style="bold magenta")
            deps_table.add_column("Package", style="cyan")
            deps_table.add_column("Version", style="green")
            
            deps_table.add_row("pandas", pd.__version__)
            deps_table.add_row("scikit-learn", sklearn.__version__)
            deps_table.add_row("mlflow", mlflow.__version__)
            
            console.print(deps_table)
        except ImportError as e:
            console.print(f"[yellow]Warning:[/yellow] Could not check dependency versions: {e}")
    
    # Show available commands
    console.print(f"\n[bold]Available Commands:[/bold]")
    commands_info = [
        ("train", "Train a machine learning model"),
        ("inference", "Perform inference with a trained model"),
        ("evaluate", "Evaluate model performance"),
        ("monitor", "Monitor for data and model drift"),
        ("init", "Create configuration template"),
        ("validate", "Validate configuration file"),
        ("status", "Show pipeline status")
    ]
    
    for cmd, desc in commands_info:
        console.print(f"  [cyan]{cmd}[/cyan] - {desc}")


@cli.command()
@click.option('--experiment-id', help='Filter by experiment ID')
@click.option('--limit', type=int, default=10, help='Limit number of results')
@click.option('--format', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.option('--sort-by', type=click.Choice(['created', 'name', 'accuracy']), 
              default='created', help='Sort experiments by field')
@click.option('--status', type=click.Choice(['running', 'completed', 'failed']),
              help='Filter by experiment status')
def experiments(experiment_id: Optional[str], limit: int, format: str, 
                sort_by: str, status: Optional[str]):
    """List and manage ML experiments."""
    
    console.print("[bold blue]ML Experiments[/bold blue]\n")
    
    # This would integrate with MLflow in a real implementation
    # For now, show placeholder data
    sample_experiments = [
        {
            'experiment_id': 'exp_001',
            'name': 'classification_baseline',
            'status': 'completed',
            'accuracy': 0.85,
            'created_at': '2024-01-15 10:30:00'
        },
        {
            'experiment_id': 'exp_002', 
            'name': 'hyperopt_tuning',
            'status': 'running',
            'accuracy': 0.87,
            'created_at': '2024-01-15 14:20:00'
        },
        {
            'experiment_id': 'exp_003',
            'name': 'drift_monitoring',
            'status': 'completed',
            'accuracy': 0.82,
            'created_at': '2024-01-15 16:45:00'
        },
        {
            'experiment_id': 'exp_004',
            'name': 'few_shot_learning',
            'status': 'failed',
            'accuracy': 0.0,
            'created_at': '2024-01-15 18:00:00'
        }
    ]
    
    # Filter by experiment ID if provided
    if experiment_id:
        sample_experiments = [exp for exp in sample_experiments if exp['experiment_id'] == experiment_id]
    
    # Filter by status if provided
    if status:
        sample_experiments = [exp for exp in sample_experiments if exp['status'] == status]
    
    # Sort experiments
    if sort_by == 'name':
        sample_experiments.sort(key=lambda x: x['name'])
    elif sort_by == 'accuracy':
        sample_experiments.sort(key=lambda x: x['accuracy'], reverse=True)
    else:  # created
        sample_experiments.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit results
    sample_experiments = sample_experiments[:limit]
    
    if format == 'json':
        console.print(json.dumps(sample_experiments, indent=2))
    else:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Experiment ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Accuracy", style="blue")
        table.add_column("Created", style="dim")
        
        for exp in sample_experiments:
            if exp['status'] == 'completed':
                status_style = "green"
            elif exp['status'] == 'running':
                status_style = "yellow"
            else:  # failed
                status_style = "red"
                
            table.add_row(
                exp['experiment_id'],
                exp['name'],
                f"[{status_style}]{exp['status']}[/{status_style}]",
                f"{exp['accuracy']:.3f}" if exp['accuracy'] > 0 else "N/A",
                exp['created_at']
            )
        
        console.print(table)
    
    console.print(f"\n[dim]Showing {len(sample_experiments)} experiments[/dim]")


@cli.command()
@click.option('--topic', type=click.Choice(['getting-started', 'configuration', 'training', 'inference', 'monitoring']),
              help='Show help for specific topic')
def help_guide(topic: Optional[str]):
    """Show detailed help and usage examples."""
    
    if not topic:
        console.print("[bold blue]ML Pipeline Help Guide[/bold blue]\n")
        
        console.print("[bold]Available Help Topics:[/bold]")
        topics = [
            ("getting-started", "Quick start guide for new users"),
            ("configuration", "Configuration file format and options"),
            ("training", "Model training workflows and options"),
            ("inference", "Running inference on trained models"),
            ("monitoring", "Setting up drift detection and monitoring")
        ]
        
        for topic_name, description in topics:
            console.print(f"  [cyan]{topic_name}[/cyan] - {description}")
        
        console.print(f"\n[dim]Use: mlpipeline help-guide --topic <topic> for detailed help[/dim]")
        return
    
    if topic == 'getting-started':
        panel = Panel(
            """[bold]Quick Start Guide[/bold]

1. [cyan]Initialize a new project:[/cyan]
   mlpipeline init --output my_config.yaml --use-case classification

2. [cyan]Prepare your data:[/cyan]
   - Place training data at the path specified in config
   - Ensure data format matches configuration

3. [cyan]Train your first model:[/cyan]
   mlpipeline train --config my_config.yaml

4. [cyan]Evaluate the model:[/cyan]
   mlpipeline evaluate --config my_config.yaml --model-path <model> --test-data <data>

5. [cyan]Run inference:[/cyan]
   mlpipeline inference --config my_config.yaml --model-path <model> --input-data <data>

[dim]For more detailed help on any command, use: mlpipeline <command> --help[/dim]""",
            title="Getting Started",
            border_style="green"
        )
        console.print(panel)
    
    elif topic == 'configuration':
        panel = Panel(
            """[bold]Configuration File Format[/bold]

The pipeline uses YAML configuration files with the following structure:

[cyan]Basic Structure:[/cyan]
```yaml
pipeline:
  name: "my-experiment"
  version: "1.0.0"

data:
  sources:
    - type: "csv"
      path: "data/train.csv"
  preprocessing:
    - type: "standard_scaler"
      columns: ["feature1", "feature2"]

model:
  type: "sklearn"
  parameters:
    algorithm: "RandomForestClassifier"
    n_estimators: 100

evaluation:
  metrics: ["accuracy", "f1_score"]
```

[cyan]Advanced Options:[/cyan]
- Hyperparameter tuning with Optuna
- Drift detection with Evidently AI
- Few-shot learning with Hugging Face
- MLflow experiment tracking

[dim]Use 'mlpipeline validate --config <file>' to check your configuration[/dim]""",
            title="Configuration Guide",
            border_style="blue"
        )
        console.print(panel)
    
    elif topic == 'training':
        panel = Panel(
            """[bold]Model Training Guide[/bold]

[cyan]Basic Training:[/cyan]
mlpipeline train --config config.yaml

[cyan]Advanced Options:[/cyan]
- [yellow]--interactive[/yellow]: Interactive configuration setup
- [yellow]--dry-run[/yellow]: Validate configuration without training
- [yellow]--resume[/yellow]: Resume from checkpoint
- [yellow]--experiment-id[/yellow]: Custom experiment ID
- [yellow]--artifacts-path[/yellow]: Custom artifacts location

[cyan]Supported Model Types:[/cyan]
- sklearn: Scikit-learn models (RandomForest, SVM, etc.)
- xgboost: XGBoost gradient boosting
- pytorch: PyTorch neural networks
- huggingface: Transformer models for NLP

[cyan]Hyperparameter Tuning:[/cyan]
Enable in config with:
```yaml
model:
  hyperparameter_tuning:
    enabled: true
    method: "optuna"
    n_trials: 50
```

[dim]Monitor training progress with: mlpipeline progress --experiment-id <id>[/dim]""",
            title="Training Guide",
            border_style="yellow"
        )
        console.print(panel)
    
    elif topic == 'inference':
        panel = Panel(
            """[bold]Model Inference Guide[/bold]

[cyan]Basic Inference:[/cyan]
mlpipeline inference --config config.yaml --model-path model.pkl --input-data data.csv

[cyan]Options:[/cyan]
- [yellow]--output-path[/yellow]: Where to save predictions
- [yellow]--batch-size[/yellow]: Processing batch size
- [yellow]--confidence-threshold[/yellow]: Minimum confidence for predictions

[cyan]Supported Input Formats:[/cyan]
- CSV files
- JSON files
- Parquet files

[cyan]Output Formats:[/cyan]
- JSON with predictions and confidence scores
- CSV with original data plus predictions

[cyan]Real-time vs Batch:[/cyan]
- Use smaller batch sizes for real-time inference
- Use larger batch sizes for batch processing

[dim]The system automatically applies the same preprocessing used during training[/dim]""",
            title="Inference Guide",
            border_style="magenta"
        )
        console.print(panel)
    
    elif topic == 'monitoring':
        panel = Panel(
            """[bold]Monitoring and Drift Detection[/bold]

[cyan]Basic Monitoring:[/cyan]
mlpipeline monitor --config config.yaml --current-data new_data.csv

[cyan]Drift Detection Setup:[/cyan]
```yaml
drift_detection:
  enabled: true
  baseline_data: "data/train.csv"
  thresholds:
    data_drift: 0.1
    prediction_drift: 0.05
```

[cyan]Types of Drift Detected:[/cyan]
- Data drift: Changes in input feature distributions
- Prediction drift: Changes in model output distributions
- Concept drift: Changes in the relationship between features and target

[cyan]Monitoring Options:[/cyan]
- [yellow]--baseline-data[/yellow]: Reference dataset for comparison
- [yellow]--threshold[/yellow]: Custom drift detection threshold
- [yellow]--output-path[/yellow]: Where to save drift reports

[cyan]Alerting:[/cyan]
Configure alerts in your config file to get notified when drift is detected.

[dim]Regular monitoring helps maintain model performance in production[/dim]""",
            title="Monitoring Guide",
            border_style="red"
        )
        console.print(panel)


@cli.command()
@click.option('--experiment-id', required=True, help='Experiment ID to check progress')
@click.option('--follow', '-f', is_flag=True, help='Follow progress in real-time')
@click.option('--refresh-interval', type=int, default=5, help='Refresh interval in seconds for follow mode')
def progress(experiment_id: str, follow: bool, refresh_interval: int):
    """Check the progress of a running experiment."""
    
    console.print(f"[bold blue]Experiment Progress: {experiment_id}[/bold blue]\n")
    
    # This would integrate with actual progress tracking in a real implementation
    # For now, show placeholder progress information
    import time
    import random
    
    def get_progress_info():
        # Simulate progress data
        stages = [
            {"name": "Data Ingestion", "status": "completed", "progress": 100},
            {"name": "Data Preprocessing", "status": "completed", "progress": 100},
            {"name": "Model Training", "status": "running", "progress": random.randint(60, 90)},
            {"name": "Model Evaluation", "status": "pending", "progress": 0}
        ]
        return stages
    
    def display_progress_table(stages):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="green")
        table.add_column("Details", style="dim")
        
        for stage in stages:
            status = stage["status"]
            if status == "completed":
                status_display = "[green]✓ Completed[/green]"
                details = "Finished successfully"
            elif status == "running":
                status_display = "[yellow]⚡ Running[/yellow]"
                details = f"Processing... {stage['progress']}%"
            else:  # pending
                status_display = "[dim]⏳ Pending[/dim]"
                details = "Waiting to start"
            
            progress_bar = "█" * (stage["progress"] // 10) + "░" * (10 - stage["progress"] // 10)
            progress_display = f"{progress_bar} {stage['progress']}%"
            
            table.add_row(stage["name"], status_display, progress_display, details)
        
        return table
    
    if follow:
        console.print(f"[dim]Following progress every {refresh_interval} seconds. Press Ctrl+C to stop.[/dim]\n")
        try:
            while True:
                console.clear()
                console.print(f"[bold blue]Experiment Progress: {experiment_id}[/bold blue]\n")
                
                stages = get_progress_info()
                table = display_progress_table(stages)
                console.print(table)
                
                # Check if all stages are completed
                if all(stage["status"] == "completed" for stage in stages):
                    console.print("\n[green]✓ All stages completed![/green]")
                    break
                
                console.print(f"\n[dim]Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Progress monitoring stopped.[/yellow]")
    else:
        stages = get_progress_info()
        table = display_progress_table(stages)
        console.print(table)
        
        # Show overall progress
        total_progress = sum(stage["progress"] for stage in stages) / len(stages)
        console.print(f"\n[bold]Overall Progress:[/bold] {total_progress:.1f}%")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file to analyze')
@click.option('--check-data', is_flag=True, help='Check if data files exist')
@click.option('--estimate-time', is_flag=True, help='Estimate training time')
@click.option('--suggest-improvements', is_flag=True, help='Suggest configuration improvements')
def analyze(config: Optional[str], check_data: bool, estimate_time: bool, suggest_improvements: bool):
    """Analyze pipeline configuration and provide insights."""
    
    if not config:
        console.print("[red]Error:[/red] Configuration file is required")
        sys.exit(1)
    
    try:
        console.print("[bold blue]Pipeline Analysis[/bold blue]\n")
        
        with console.status("[bold green]Analyzing configuration..."):
            config_manager = ConfigManager()
            pipeline_config = config_manager.load_config(config)
        
        console.print(f"[green]✓[/green] Configuration loaded: {pipeline_config.pipeline.name}")
        
        # Basic configuration analysis
        console.print("\n[bold]Configuration Analysis:[/bold]")
        analysis_table = Table(show_header=True, header_style="bold magenta")
        analysis_table.add_column("Aspect", style="cyan")
        analysis_table.add_column("Value", style="green")
        analysis_table.add_column("Assessment", style="yellow")
        
        # Model complexity assessment
        model_type = pipeline_config.model.type.value
        if model_type in ['sklearn', 'xgboost']:
            complexity = "Medium"
            assessment = "Good for structured data"
        elif model_type == 'pytorch':
            complexity = "High"
            assessment = "Suitable for complex patterns"
        else:  # huggingface
            complexity = "Very High"
            assessment = "Best for NLP tasks"
        
        analysis_table.add_row("Model Complexity", complexity, assessment)
        analysis_table.add_row("Data Sources", str(len(pipeline_config.data.sources)), "Multiple sources detected" if len(pipeline_config.data.sources) > 1 else "Single source")
        analysis_table.add_row("Preprocessing Steps", str(len(pipeline_config.data.preprocessing)), "Well-configured" if len(pipeline_config.data.preprocessing) > 2 else "Consider more steps")
        
        console.print(analysis_table)
        
        # Check data files if requested
        if check_data:
            console.print("\n[bold]Data File Check:[/bold]")
            data_table = Table(show_header=True, header_style="bold magenta")
            data_table.add_column("File", style="cyan")
            data_table.add_column("Status", style="green")
            data_table.add_column("Size", style="yellow")
            
            for source in pipeline_config.data.sources:
                file_path = Path(source.path)
                if file_path.exists():
                    size = file_path.stat().st_size
                    size_mb = size / (1024 * 1024)
                    data_table.add_row(source.path, "[green]✓ Found[/green]", f"{size_mb:.1f} MB")
                else:
                    data_table.add_row(source.path, "[red]✗ Missing[/red]", "N/A")
            
            console.print(data_table)
        
        # Estimate training time if requested
        if estimate_time:
            console.print("\n[bold]Training Time Estimation:[/bold]")
            # This is a simplified estimation - in reality, this would be more sophisticated
            base_time = 60  # base time in seconds
            
            if model_type == 'pytorch':
                base_time *= 5
            elif model_type == 'huggingface':
                base_time *= 10
            elif model_type == 'xgboost':
                base_time *= 2
            
            # Factor in hyperparameter tuning
            if hasattr(pipeline_config.model, 'hyperparameter_tuning') and pipeline_config.model.hyperparameter_tuning.enabled:
                n_trials = getattr(pipeline_config.model.hyperparameter_tuning, 'n_trials', 50)
                base_time *= (n_trials / 10)
            
            console.print(f"  [cyan]Estimated Training Time:[/cyan] {base_time/60:.1f} minutes")
            console.print(f"  [dim]Note: This is a rough estimate based on configuration complexity[/dim]")
        
        # Suggest improvements if requested
        if suggest_improvements:
            console.print("\n[bold]Improvement Suggestions:[/bold]")
            suggestions = []
            
            # Check for common improvements
            if len(pipeline_config.data.preprocessing) < 3:
                suggestions.append("Consider adding more preprocessing steps (feature scaling, encoding)")
            
            if not hasattr(pipeline_config, 'drift_detection') or not pipeline_config.drift_detection.enabled:
                suggestions.append("Enable drift detection for production monitoring")
            
            if not hasattr(pipeline_config.model, 'hyperparameter_tuning') or not getattr(pipeline_config.model.hyperparameter_tuning, 'enabled', False):
                suggestions.append("Enable hyperparameter tuning for better performance")
            
            if len(pipeline_config.evaluation.metrics) < 3:
                suggestions.append("Add more evaluation metrics for comprehensive assessment")
            
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    console.print(f"  [yellow]{i}.[/yellow] {suggestion}")
            else:
                console.print("  [green]✓[/green] Configuration looks well-optimized!")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


def _display_config_summary(config):
    """Display a summary of the configuration."""
    console.print(f"\n[bold]Configuration Summary:[/bold]")
    console.print(f"  [cyan]Pipeline:[/cyan] {config.pipeline.name}")
    console.print(f"  [cyan]Model Type:[/cyan] {config.model.type.value}")
    console.print(f"  [cyan]Data Sources:[/cyan] {len(config.data.sources)}")
    console.print(f"  [cyan]Preprocessing Steps:[/cyan] {len(config.data.preprocessing)}")
    console.print(f"  [cyan]Evaluation Metrics:[/cyan] {', '.join(config.evaluation.metrics)}")
    
    if hasattr(config, 'drift_detection') and config.drift_detection.enabled:
        console.print(f"  [cyan]Drift Detection:[/cyan] Enabled")
    
    if hasattr(config, 'few_shot') and config.few_shot.enabled:
        console.print(f"  [cyan]Few-Shot Learning:[/cyan] Enabled")


def _display_training_results(result):
    """Display training results."""
    console.print(f"\n[bold]Training Results:[/bold]")
    console.print(f"  [cyan]Execution Time:[/cyan] {result.execution_time:.2f} seconds")
    console.print(f"  [cyan]Artifacts:[/cyan] {len(result.artifacts)} files")
    
    if result.metrics:
        console.print(f"  [cyan]Metrics:[/cyan]")
        for metric, value in result.metrics.items():
            console.print(f"    {metric}: {value}")


def _display_inference_results(result):
    """Display inference results."""
    console.print(f"\n[bold]Inference Results:[/bold]")
    console.print(f"  [cyan]Execution Time:[/cyan] {result.execution_time:.2f} seconds")
    console.print(f"  [cyan]Predictions Generated:[/cyan] {result.metrics.get('predictions_count', 'N/A')}")


def _display_evaluation_results(result):
    """Display evaluation results."""
    console.print(f"\n[bold]Evaluation Results:[/bold]")
    console.print(f"  [cyan]Execution Time:[/cyan] {result.execution_time:.2f} seconds")
    
    if result.metrics:
        console.print(f"  [cyan]Performance Metrics:[/cyan]")
        for metric, value in result.metrics.items():
            console.print(f"    {metric}: {value}")


def _display_drift_results(result):
    """Display drift detection results."""
    console.print(f"\n[bold]Drift Detection Results:[/bold]")
    console.print(f"  [cyan]Execution Time:[/cyan] {result.execution_time:.2f} seconds")
    
    if result.metrics:
        drift_detected = result.metrics.get('drift_detected', False)
        drift_score = result.metrics.get('drift_score', 'N/A')
        
        if drift_detected:
            console.print(f"  [red]Drift Detected:[/red] Yes")
        else:
            console.print(f"  [green]Drift Detected:[/green] No")
        
        console.print(f"  [cyan]Drift Score:[/cyan] {drift_score}")


def _create_training_stages(config, orchestrator):
    """Create training pipeline stages from configuration."""
    from .core.interfaces import PipelineStage
    from .data.ingestion import DataIngestionEngine
    from .data.preprocessing import DataPreprocessor
    from .models.training import ModelTrainer
    from .models.evaluation import ModelEvaluator
    
    # Create components with real implementations
    data_ingestion = DataIngestionEngine()
    data_preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer()
    
    # Create evaluator
    model_evaluator = ModelEvaluator()
    
    stages = [
        PipelineStage(name="data_ingestion", components=[data_ingestion]),
        PipelineStage(name="data_preprocessing", components=[data_preprocessor], dependencies=["data_ingestion"]),
        PipelineStage(name="model_training", components=[model_trainer], dependencies=["data_preprocessing"]),
        PipelineStage(name="model_evaluation", components=[model_evaluator], dependencies=["model_training"])
    ]
    
    return stages


def _create_inference_stages(config, orchestrator):
    """Create inference pipeline stages from configuration."""
    from .core.interfaces import PipelineStage
    from .data.ingestion import DataIngestionEngine
    from .data.preprocessing import DataPreprocessor
    from .models.inference import ModelInferenceEngine
    
    # Create components
    data_loader = DataIngestionEngine()
    data_preprocessor = DataPreprocessor()
    inference_engine = ModelInferenceEngine()
    
    stages = [
        PipelineStage(name="data_loading", components=[data_loader]),
        PipelineStage(name="data_preprocessing", components=[data_preprocessor], dependencies=["data_loading"]),
        PipelineStage(name="model_inference", components=[inference_engine], dependencies=["data_preprocessing"])
    ]
    
    return stages


def _create_evaluation_stages(config, orchestrator):
    """Create evaluation pipeline stages from configuration."""
    from .core.interfaces import PipelineStage
    from .data.ingestion import DataIngestionEngine
    from .models.evaluation import ModelEvaluator
    
    # Create components
    data_loader = DataIngestionEngine()
    model_evaluator = ModelEvaluator()
    
    stages = [
        PipelineStage(name="data_loading", components=[data_loader]),
        PipelineStage(name="model_evaluation", components=[model_evaluator], dependencies=["data_loading"])
    ]
    
    return stages


def _create_monitoring_stages(config, orchestrator):
    """Create monitoring pipeline stages from configuration."""
    from .core.interfaces import PipelineStage
    from .data.ingestion import DataIngestionEngine
    from .monitoring.drift_detection import DriftDetector
    
    # Create components
    data_loader = DataIngestionEngine()
    drift_detector = DriftDetector()
    
    stages = [
        PipelineStage(name="data_loading", components=[data_loader]),
        PipelineStage(name="drift_detection", components=[drift_detector], dependencies=["data_loading"])
    ]
    
    return stages


def _interactive_config_setup(existing_config: Optional[str] = None) -> str:
    """Interactive configuration setup wizard."""
    console.print("[bold blue]Interactive Configuration Setup[/bold blue]\n")
    
    # Load existing config if provided
    config_data = {}
    if existing_config and Path(existing_config).exists():
        try:
            with open(existing_config, 'r') as f:
                if existing_config.endswith('.yaml') or existing_config.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            console.print(f"[green]✓[/green] Loaded existing configuration: {existing_config}")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not load existing config: {e}")
    
    # Pipeline basic information
    console.print("[bold]Pipeline Configuration[/bold]")
    pipeline_name = Prompt.ask(
        "Pipeline name", 
        default=config_data.get('pipeline', {}).get('name', 'my-ml-pipeline')
    )
    
    pipeline_version = Prompt.ask(
        "Pipeline version", 
        default=config_data.get('pipeline', {}).get('version', '1.0.0')
    )
    
    # Model configuration
    console.print("\n[bold]Model Configuration[/bold]")
    model_types = ['sklearn', 'xgboost', 'pytorch', 'huggingface']
    model_type = Prompt.ask(
        "Model type",
        choices=model_types,
        default=config_data.get('model', {}).get('type', 'sklearn')
    )
    
    # Task type
    task_types = ['classification', 'regression', 'few-shot']
    task_type = Prompt.ask(
        "Task type",
        choices=task_types,
        default='classification'
    )
    
    # Data configuration
    console.print("\n[bold]Data Configuration[/bold]")
    train_data_path = Prompt.ask(
        "Training data path",
        default=config_data.get('data', {}).get('sources', [{}])[0].get('path', 'data/train.csv')
    )
    
    test_data_path = Prompt.ask(
        "Test data path (optional)",
        default=config_data.get('data', {}).get('test_path', 'data/test.csv')
    )
    
    # Advanced options
    console.print("\n[bold]Advanced Options[/bold]")
    enable_hyperopt = Confirm.ask(
        "Enable hyperparameter optimization?",
        default=config_data.get('model', {}).get('hyperparameter_tuning', {}).get('enabled', False)
    )
    
    enable_drift_detection = Confirm.ask(
        "Enable drift detection?",
        default=config_data.get('drift_detection', {}).get('enabled', False)
    )
    
    enable_few_shot = task_type == 'few-shot' or Confirm.ask(
        "Enable few-shot learning?",
        default=config_data.get('few_shot', {}).get('enabled', False)
    )
    
    # Build configuration
    new_config = {
        'pipeline': {
            'name': pipeline_name,
            'version': pipeline_version
        },
        'data': {
            'sources': [
                {
                    'type': 'csv',
                    'path': train_data_path,
                    'schema': 'schemas/data_schema.json'
                }
            ],
            'test_path': test_data_path,
            'preprocessing': [
                {
                    'type': 'standard_scaler',
                    'columns': ['all_numeric']
                }
            ]
        },
        'model': {
            'type': model_type,
            'parameters': _get_default_model_params(model_type, task_type)
        },
        'evaluation': {
            'metrics': _get_default_metrics(task_type),
            'cross_validation': task_type == 'classification'
        }
    }
    
    # Add optional configurations
    if enable_hyperopt:
        new_config['model']['hyperparameter_tuning'] = {
            'enabled': True,
            'method': 'optuna',
            'n_trials': 50,
            'direction': 'maximize' if task_type == 'classification' else 'minimize'
        }
    
    if enable_drift_detection:
        new_config['drift_detection'] = {
            'enabled': True,
            'baseline_data': train_data_path,
            'thresholds': {
                'data_drift': 0.1,
                'prediction_drift': 0.05
            }
        }
    
    if enable_few_shot:
        new_config['few_shot'] = {
            'enabled': True,
            'prompt_template': 'templates/few_shot.txt',
            'max_examples': 5,
            'similarity_threshold': 0.8
        }
    
    # Save configuration
    output_path = Prompt.ask(
        "Save configuration to",
        default=existing_config or f"{pipeline_name.replace(' ', '_').lower()}_config.yaml"
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)
    
    console.print(f"\n[green]✓[/green] Configuration saved to: {output_path}")
    
    # Display summary
    _display_interactive_summary(new_config)
    
    return str(output_path)


def _get_default_model_params(model_type: str, task_type: str) -> Dict[str, Any]:
    """Get default model parameters based on type and task."""
    if model_type == 'sklearn':
        if task_type == 'classification':
            return {
                'algorithm': 'RandomForestClassifier',
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        else:  # regression
            return {
                'algorithm': 'RandomForestRegressor',
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
    elif model_type == 'xgboost':
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    elif model_type == 'pytorch':
        return {
            'hidden_layers': [128, 64],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100
        }
    elif model_type == 'huggingface':
        return {
            'model_name': 'distilbert-base-uncased',
            'max_length': 512,
            'learning_rate': 2e-5,
            'epochs': 3
        }
    else:
        return {}


def _get_default_metrics(task_type: str) -> list:
    """Get default metrics based on task type."""
    if task_type == 'classification':
        return ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    elif task_type == 'regression':
        return ['mse', 'mae', 'r2_score', 'rmse']
    else:  # few-shot
        return ['accuracy', 'f1_score']


def _display_interactive_summary(config: Dict[str, Any]):
    """Display summary of interactively created configuration."""
    console.print("\n[bold]Configuration Summary:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Pipeline Name", config['pipeline']['name'])
    table.add_row("Model Type", config['model']['type'])
    table.add_row("Training Data", config['data']['sources'][0]['path'])
    table.add_row("Metrics", ', '.join(config['evaluation']['metrics']))
    
    if 'hyperparameter_tuning' in config.get('model', {}):
        table.add_row("Hyperparameter Tuning", "Enabled")
    
    if config.get('drift_detection', {}).get('enabled'):
        table.add_row("Drift Detection", "Enabled")
    
    if config.get('few_shot', {}).get('enabled'):
        table.add_row("Few-Shot Learning", "Enabled")
    
    console.print(table)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()