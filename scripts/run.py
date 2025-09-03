# Main script
"""CLI entry point for ProfLLM"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the profllm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from profllm import Experiment, ExperimentConfig
from profllm.utils.gpu import get_gpu_info, check_gpu_compatibility

console = Console()

def setup_logging(verbose: bool = False):
    """Setup rich logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """ProfLLM - Professional vLLM Benchmarking Suite"""
    setup_logging(verbose)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to experiment configuration file')
@click.option('--output', '-o', help='Output file path for results')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running experiment')
def run(config: str, output: Optional[str], dry_run: bool):
    """Run a benchmark experiment (server + client together)"""
    asyncio.run(_run_experiment(config, output, dry_run))

@cli.command()
@click.option('--config', '-c', required=True, help='Path to experiment configuration file')
@click.option('--dry-run', is_flag=True, help='Show what would be executed without running')
def server(config: str, dry_run: bool):
    """Start vLLM server only"""
    asyncio.run(_run_server(config, dry_run))

@cli.command()
@click.option('--config', '-c', required=True, help='Path to experiment configuration file')
@click.option('--server-url', required=True, help='Server URL (e.g., http://127.0.0.1:8000)')
@click.option('--dry-run', is_flag=True, help='Show what would be executed without running')
def client(config: str, server_url: str, dry_run: bool):
    """Run benchmark client only (requires running server)"""
    asyncio.run(_run_client(config, server_url, dry_run))

async def _run_experiment(config_path: str, output_path: Optional[str], dry_run: bool):
    """Execute the experiment"""
    try:
        # Load configuration
        console.print(f"[blue]Loading configuration from {config_path}[/blue]")
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = ExperimentConfig.from_yaml(config_path)
        else:
            config = ExperimentConfig.from_json(config_path)
        
        # Override output path if provided
        if output_path:
            config.output_path = output_path
        
        # Set dry run flag
        config.dry_run = dry_run
        
        # Display configuration summary
        _display_config_summary(config)
        
        if dry_run:
            console.print("[yellow]Dry run mode - configuration validation only[/yellow]")
            
            # Validate configuration
            from profllm.config.validation import validate_config
            warnings = validate_config(config)
            
            if warnings:
                console.print("\n[yellow]Configuration warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")
            else:
                console.print("[green]✓ Configuration validation passed[/green]")
            
            return
        
        # Create and run experiment
        experiment = Experiment(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiment...", total=None)
            
            try:
                result = await experiment.run()
                progress.update(task, description="✓ Experiment completed")
                
                # Display results summary
                _display_results_summary(result)
                
            except Exception as e:
                progress.update(task, description="✗ Experiment failed")
                console.print(f"[red]Experiment failed: {str(e)}[/red]")
                raise
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

async def _run_server(config_path: str, dry_run: bool):
    """Start vLLM server only"""
    try:
        # Load configuration
        console.print(f"[blue]Loading server configuration from {config_path}[/blue]")
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = ExperimentConfig.from_yaml(config_path)
        else:
            config = ExperimentConfig.from_json(config_path)
        
        # Display server configuration
        console.print(f"[green]Starting vLLM server for model: {config.server.model}[/green]")
        console.print(f"[green]Tensor parallel size: {config.server.tensor_parallel_size}[/green]")
        console.print(f"[green]Port: {config.server.port or 'auto-assigned'}[/green]")
        
        if dry_run:
            console.print("[yellow]Dry run mode - server configuration only[/yellow]")
            return
        
        # Start server only
        from profllm.core.server import VLLMServerManager
        server_manager = VLLMServerManager(config.server)
        
        try:
            await server_manager.start()
            console.print(f"[green]✓ vLLM server started successfully on port {server_manager.port}[/green]")
            console.print(f"[green]Server URL: http://{server_manager.config.host}:{server_manager.port}[/green]")
            console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping server...[/yellow]")
            await server_manager.stop()
            console.print("[green]✓ Server stopped[/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

async def _run_client(config_path: str, server_url: str, dry_run: bool):
    """Run benchmark client only"""
    try:
        # Load configuration
        console.print(f"[blue]Loading client configuration from {config_path}[/blue]")
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = ExperimentConfig.from_yaml(config_path)
        else:
            config = ExperimentConfig.from_json(config_path)
        
        # Parse server URL
        if not server_url.startswith('http'):
            server_url = f"http://{server_url}"
        
        # Extract host and port from server URL
        from urllib.parse import urlparse
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port
        
        
        console.print(f"[green]Running benchmark against server: {server_url}[/green]")
        console.print(f"[green]Dataset: {config.benchmark.dataset_name}[/green]")
        console.print(f"[green]Number of prompts: {config.benchmark.num_prompts}[/green]")
        
        if dry_run:
            console.print("[yellow]Dry run mode - client configuration only[/yellow]")
            return
        
        # Create benchmark client
        from profllm.core.client import BenchmarkClient
        client = BenchmarkClient(config.benchmark, config.server)
        
        # Update server info
        client.update_server_info(host, port)
        
        # Run benchmark
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=None)
            
            try:
                result = await client.run()
                progress.update(task, description="✓ Benchmark completed")
                
                # Display results summary
                _display_results_summary(result)
                
            except Exception as e:
                progress.update(task, description="✗ Benchmark failed")
                console.print(f"[red]Benchmark failed: {str(e)}[/red]")
                raise
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

def _display_config_summary(config: ExperimentConfig):
    """Display configuration summary"""
    table = Table(title="Experiment Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    # Basic info
    table.add_row("Suite", config.suite)
    table.add_row("Model", config.server.model)
    table.add_row("Tensor Parallel Size", str(config.server.tensor_parallel_size))
    table.add_row("Dataset", config.benchmark.dataset_name)
    table.add_row("Number of Prompts", str(config.benchmark.num_prompts))
    table.add_row("Request Rate", str(config.benchmark.request_rate))
    
    console.print(table)

def _display_results_summary(result):
    """Display experiment results summary"""
    table = Table(title="Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    # Handle both experiment result objects and benchmark result dictionaries
    if hasattr(result, 'results'):
        # Full experiment result object
        results = result.results
        table.add_row("Status", result.status)
        table.add_row("Duration (s)", str(result.duration))
        
        # System metrics
        if hasattr(result, 'system') and result.system:
            table.add_row("Peak GPU Memory", f"{result.system.get('peak_gpu_memory_gb', 0):.2f} GB")
            table.add_row("Avg GPU Utilization", f"{result.system.get('avg_gpu_utilization', 0):.1f}%")
    else:
        # Benchmark result dictionary
        results = result
        table.add_row("Duration (s)", str(results.get('duration', 0)))
    
    # Performance metrics (common to both)
    table.add_row("Request Throughput", f"{results.get('request_throughput', 0):.2f} req/s")
    table.add_row("Output Throughput", f"{results.get('output_throughput', 0):.2f} tokens/s")
    table.add_row("Median TTFT", f"{results.get('median_ttft_ms', 0):.2f} ms")
    table.add_row("P95 TTFT", f"{results.get('p95_ttft_ms', 0):.2f} ms")
    table.add_row("Completed Requests", str(results.get('completed_requests', 0)))
    table.add_row("Failed Requests", str(results.get('failed_requests', 0)))
    
    console.print(table)

@cli.command()
def info():
    """Display system information"""
    from profllm.utils.system import get_system_info, format_bytes
    
    console.print("[blue]System Information[/blue]")
    
    system_info = get_system_info()
    
    # Platform info
    platform_table = Table(title="Platform")
    platform_table.add_column("Property", style="cyan")
    platform_table.add_column("Value", style="white")
    
    for key, value in system_info['platform'].items():
        platform_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(platform_table)
    
    # CPU info
    cpu_table = Table(title="CPU")
    cpu_table.add_column("Property", style="cyan")
    cpu_table.add_column("Value", style="white")
    
    for key, value in system_info['cpu'].items():
        cpu_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(cpu_table)
    
    # Memory info
    memory_table = Table(title="Memory")
    memory_table.add_column("Property", style="cyan")
    memory_table.add_column("Value", style="white")
    
    for key, value in system_info['memory'].items():
        if isinstance(value, int) and key != 'percent':
            memory_table.add_row(key.replace('_', ' ').title(), format_bytes(value))
        else:
            memory_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(memory_table)
    
    # GPU info
    if system_info['gpus']:
        gpu_table = Table(title="GPUs")
        gpu_table.add_column("Index", style="cyan")
        gpu_table.add_column("Name", style="white")
        gpu_table.add_column("Memory", style="white")
        gpu_table.add_column("Utilization", style="white")
        gpu_table.add_column("Temperature", style="white")
        
        for gpu in system_info['gpus']:
            memory_total = format_bytes(gpu.get('memory_total', 0))
            memory_used = format_bytes(gpu.get('memory_used', 0))
            gpu_table.add_row(
                str(gpu['index']),
                gpu['name'],
                f"{memory_used} / {memory_total}",
                f"{gpu.get('utilization_gpu', 0)}%",
                f"{gpu.get('temperature', 0)}°C"
            )
        
        console.print(gpu_table)
    else:
        console.print("[yellow]No GPU information available[/yellow]")

@cli.command()
@click.option('--tensor-parallel-size', '-tp', default=1, help='Tensor parallel size to check')
@click.option('--model-size-gb', type=float, help='Estimated model size in GB')
def check_gpu(tensor_parallel_size: int, model_size_gb: Optional[float]):
    """Check GPU compatibility for given configuration"""
    from profllm.utils.gpu import check_gpu_compatibility
    
    result = check_gpu_compatibility(tensor_parallel_size, model_size_gb)
    
    table = Table(title="GPU Compatibility Check")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Compatible", "✓ Yes" if result['compatible'] else "✗ No")
    table.add_row("Total GPUs", str(result['total_gpus']))
    table.add_row("Total Memory", f"{result['total_memory_gb']:.1f} GB")
    table.add_row("Memory per GPU", f"{result['memory_per_gpu_gb']:.1f} GB")
    
    console.print(table)
    
    if result['warnings']:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result['warnings']:
            console.print(f"  • {warning}")

@cli.command()
@click.argument('config_file')
def validate(config_file: str):
    """Validate experiment configuration"""
    try:
        # Load configuration
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config = ExperimentConfig.from_yaml(config_file)
        else:
            config = ExperimentConfig.from_json(config_file)
        
        # Validate
        from profllm.config.validation import validate_config
        warnings = validate_config(config)
        
        if warnings:
            console.print("[yellow]Configuration warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")
        else:
            console.print("[green]✓ Configuration is valid[/green]")
            
    except Exception as e:
        console.print(f"[red]Configuration validation failed: {str(e)}[/red]")
        sys.exit(1)

def main():
    """Main entry point"""
    cli()

if __name__ == '__main__':
    main()

