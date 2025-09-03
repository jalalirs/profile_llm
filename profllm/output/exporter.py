"""Export profllm results to various formats"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class ResultExporter:
    """Export benchmark results to various formats"""
    
    def __init__(self, config):
        self.config = config
        self.export_config = config.export
    
    def export_results(self, result_data: Dict[str, Any], run_id: str, results_dir: Path) -> Dict[str, str]:
        """Export results to configured formats with performance monitoring"""
        if not self.export_config.enable_export:
            return {}
        
        import time
        start_time = time.time()
        export_paths = {}
        
        # Create export directory
        export_dir = results_dir / self.export_config.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export CSV if enabled
        if self.export_config.export_csv:
            csv_start = time.time()
            csv_paths = self._export_to_csv(result_data, run_id, export_dir)
            export_paths.update(csv_paths)
            csv_time = time.time() - csv_start
            logger.info(f"CSV export completed in {csv_time:.2f}s")
        
        # Export InfluxDB if enabled
        if self.export_config.export_influxdb:
            influx_start = time.time()
            influx_path = self._export_to_influxdb(result_data, run_id, export_dir)
            if influx_path:
                export_paths['influxdb'] = str(influx_path)
            influx_time = time.time() - influx_start
            logger.info(f"InfluxDB export completed in {influx_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Total export completed in {total_time:.2f}s")
        
        return export_paths
    
    def _export_to_csv(self, data: Dict[str, Any], run_id: str, export_dir: Path) -> Dict[str, str]:
        """Export to CSV format"""
        csv_paths = {}
        
        try:
            # Convert data to CSV format
            summary_rows, request_rows, system_rows = self._convert_to_csv_format(data)
            
            # Save summary metrics
            if self.export_config.csv_summary and summary_rows:
                summary_path = export_dir / f"summary_metrics_{run_id}.csv"
                self._save_csv(summary_rows, summary_path, self._get_summary_fieldnames())
                csv_paths['summary_csv'] = str(summary_path)
                logger.info(f"Summary CSV exported to: {summary_path}")
            
            # Save request details
            if self.export_config.csv_requests and request_rows:
                request_path = export_dir / f"request_details_{run_id}.csv"
                self._save_csv(request_rows, request_path, self._get_request_fieldnames())
                csv_paths['request_csv'] = str(request_path)
                logger.info(f"Request CSV exported to: {request_path}")
            
            # Save system metrics
            if self.export_config.csv_system and system_rows:
                system_path = export_dir / f"system_metrics_{run_id}.csv"
                self._save_csv(system_rows, system_path, self._get_system_fieldnames())
                csv_paths['system_csv'] = str(system_path)
                logger.info(f"System CSV exported to: {system_path}")
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
        
        return csv_paths
    
    def _convert_to_csv_format(self, data: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Convert data to CSV format"""
        run_id = data['run_id']
        model = data['config']['server']['model']
        suite = data.get('suite', 'unknown')
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        # Summary metrics (one row per run)
        summary_rows = [{
            'run_id': run_id,
            'model': model,
            'suite': suite,
            'timestamp': timestamp.isoformat(),
            'duration': data['results']['duration'],
            'request_throughput': data['results']['request_throughput'],
            'output_throughput': data['results']['output_throughput'],
            'total_token_throughput': data['results']['raw_data']['total_token_throughput'],
            'median_ttft_ms': data['results']['median_ttft_ms'],
            'p95_ttft_ms': data['results']['p95_ttft_ms'],
            'p99_ttft_ms': data['results']['raw_data']['p99_ttft_ms'],
            'median_tpot_ms': data['results']['median_tpot_ms'],
            'p95_tpot_ms': data['results']['p95_tpot_ms'],
            'p99_tpot_ms': data['results']['raw_data']['p99_tpot_ms'],
            'median_itl_ms': data['results']['raw_data']['median_itl_ms'],
            'p95_itl_ms': data['results']['raw_data']['p95_itl_ms'],
            'p99_itl_ms': data['results']['raw_data']['p99_itl_ms'],
            'total_requests': data['results']['completed_requests'],
            'failed_requests': data['results']['failed_requests'],
            'total_input_tokens': data['results']['total_input_tokens'],
            'total_output_tokens': data['results']['total_output_tokens'],
            'mean_ttft_ms': data['results']['raw_data']['mean_ttft_ms'],
            'std_ttft_ms': data['results']['raw_data']['std_ttft_ms'],
            'mean_tpot_ms': data['results']['raw_data']['mean_tpot_ms'],
            'std_tpot_ms': data['results']['raw_data']['std_tpot_ms'],
            'mean_itl_ms': data['results']['raw_data']['mean_itl_ms'],
            'std_itl_ms': data['results']['raw_data']['std_itl_ms']
        }]
        
        # Request-level details - using generator for memory efficiency
        def _generate_request_rows():
            if 'raw_data' in data['results']:
                raw_data = data['results']['raw_data']
                input_lens = raw_data.get('input_lens', [])
                output_lens = raw_data.get('output_lens', [])
                ttfts = raw_data.get('ttfts', [])
                itls = raw_data.get('itls', [])
                generated_texts = raw_data.get('generated_texts', [])
                errors = raw_data.get('errors', [])
                
                for i in range(len(input_lens)):
                    yield {
                        'run_id': run_id,
                        'model': model,
                        'suite': suite,
                        'timestamp': timestamp.isoformat(),
                        'request_id': i,
                        'input_tokens': input_lens[i] if i < len(input_lens) else 0,
                        'output_tokens': output_lens[i] if i < len(output_lens) else 0,
                        'ttft_ms': ttfts[i] * 1000 if i < len(ttfts) else 0,
                        'itl_ms': itls[i] * 1000 if i < len(itls) else 0,
                        'generated_text_length': len(generated_texts[i]) if i < len(generated_texts) else 0,
                        'error': errors[i] if i < len(errors) else '',
                        'request_throughput': data['results']['request_throughput'],
                        'output_throughput': data['results']['output_throughput']
                    }
        
        request_rows = list(_generate_request_rows())
        
        # System metrics time series - using generator for memory efficiency
        def _generate_system_rows():
            if 'system' in data and 'raw_metrics' in data['system']:
                for raw_metric in data['system']['raw_metrics']:
                    metric_time = datetime.fromtimestamp(raw_metric['timestamp'])
                    yield {
                        'run_id': run_id,
                        'model': model,
                        'suite': suite,
                        'timestamp': metric_time.isoformat(),
                        'cpu_percent': raw_metric['cpu_percent'],
                        'cpu_count': raw_metric['cpu_count'],
                        'cpu_count_logical': raw_metric['cpu_count_logical'],
                        'memory_total_gb': raw_metric['memory_total_gb'],
                        'memory_available_gb': raw_metric['memory_available_gb'],
                        'memory_used_gb': raw_metric['memory_used_gb'],
                        'memory_percent': raw_metric['memory_percent'],
                        'load_avg_1min': raw_metric['load_avg'][0] if raw_metric['load_avg'] else 0,
                        'load_avg_5min': raw_metric['load_avg'][1] if raw_metric['load_avg'] else 0,
                        'load_avg_15min': raw_metric['load_avg'][2] if raw_metric['load_avg'] else 0,
                        'peak_cpu_percent': data['system']['peak_cpu_percent'],
                        'avg_cpu_percent': data['system']['avg_cpu_percent'],
                        'peak_memory_gb': data['system']['peak_memory_gb'],
                        'avg_memory_gb': data['system']['avg_memory_gb'],
                        'peak_gpu_memory_gb': data['system']['peak_gpu_memory_gb'],
                        'avg_gpu_utilization': data['system']['avg_gpu_utilization']
                    }
        
        system_rows = list(_generate_system_rows())
        
        return summary_rows, request_rows, system_rows
    
    def _export_to_influxdb(self, data: Dict[str, Any], run_id: str, export_dir: Path) -> Optional[Path]:
        """Export to InfluxDB line protocol format"""
        try:
            influx_data = self._convert_to_influxdb_format(data)
            influx_path = export_dir / f"influxdb_{run_id}.txt"
            
            with open(influx_path, 'w') as f:
                f.write(influx_data)
            
            logger.info(f"InfluxDB format exported to: {influx_path}")
            return influx_path
            
        except Exception as e:
            logger.error(f"Error exporting to InfluxDB: {e}")
            return None
    
    def _convert_to_influxdb_format(self, data: Dict[str, Any]) -> str:
        """Convert to InfluxDB line protocol format"""
        lines = []
        
        # Parse timestamp
        ts_str = data['timestamp'].replace('Z', '+00:00')
        timestamp = int(datetime.fromisoformat(ts_str).timestamp() * 1000000000)
        
        model = data['config']['server']['model']
        run_id = data['run_id']
        suite = data.get('suite', 'unknown')
        
        # Summary metrics
        summary_metrics = {
            'request_throughput': data['results']['request_throughput'],
            'output_throughput': data['results']['output_throughput'],
            'total_token_throughput': data['results']['raw_data']['total_token_throughput'],
            'median_ttft_ms': data['results']['median_ttft_ms'],
            'p95_ttft_ms': data['results']['p95_ttft_ms'],
            'p99_ttft_ms': data['results']['raw_data']['p99_ttft_ms'],
            'median_tpot_ms': data['results']['median_tpot_ms'],
            'p95_tpot_ms': data['results']['p95_tpot_ms'],
            'p99_tpot_ms': data['results']['raw_data']['p99_tpot_ms'],
            'total_requests': data['results']['completed_requests'],
            'failed_requests': data['results']['failed_requests'],
            'total_input_tokens': data['results']['total_input_tokens'],
            'total_output_tokens': data['results']['total_output_tokens']
        }
        
        for metric, value in summary_metrics.items():
            if isinstance(value, (int, float)):
                line = f"benchmark_summary,model={model},run_id={run_id},suite={suite} {metric}={value} {timestamp}"
                lines.append(line)
        
        # Request-level metrics
        if 'raw_data' in data['results']:
            raw_data = data['results']['raw_data']
            input_lens = raw_data.get('input_lens', [])
            output_lens = raw_data.get('output_lens', [])
            ttfts = raw_data.get('ttfts', [])
            itls = raw_data.get('itls', [])
            
            for i in range(len(input_lens)):
                req_timestamp = timestamp + (i * 1000000000)
                line = f"request_metrics,model={model},run_id={run_id},suite={suite},request_id={i} input_tokens={input_lens[i]},output_tokens={output_lens[i]},ttft_sec={ttfts[i]},itl_sec={itls[i]} {req_timestamp}"
                lines.append(line)
        
        # System metrics
        if 'system' in data and 'raw_metrics' in data['system']:
            for raw_metric in data['system']['raw_metrics']:
                metric_time = int(raw_metric['timestamp'] * 1000000000)
                line = f"system_metrics,model={model},run_id={run_id},suite={suite} cpu_percent={raw_metric['cpu_percent']},memory_used_gb={raw_metric['memory_used_gb']},memory_percent={raw_metric['memory_percent']},load_avg_1min={raw_metric['load_avg'][0]},load_avg_5min={raw_metric['load_avg'][1]},load_avg_15min={raw_metric['load_avg'][2]} {metric_time}"
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _save_csv(self, rows: List[Dict], filepath: Path, fieldnames: List[str]):
        """Save data to CSV file with optimized batch writing"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Use writerows for batch writing instead of individual writes
            writer.writerows(rows)
    
    def _save_csv_streaming(self, data_generator, filepath: Path, fieldnames: List[str], chunk_size: int = 1000):
        """Save data to CSV file using streaming for very large datasets"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process data in chunks to minimize memory usage
            chunk = []
            for item in data_generator:
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    writer.writerows(chunk)
                    chunk = []
            
            # Write remaining items
            if chunk:
                writer.writerows(chunk)
    
    @lru_cache(maxsize=32)
    def _get_summary_fieldnames(self) -> List[str]:
        """Get fieldnames for summary CSV (cached for performance)"""
        return [
            'run_id', 'model', 'suite', 'timestamp', 'duration', 'request_throughput', 
            'output_throughput', 'total_token_throughput', 'median_ttft_ms', 'p95_ttft_ms', 
            'p99_ttft_ms', 'median_tpot_ms', 'p95_tpot_ms', 'p99_tpot_ms', 'median_itl_ms', 
            'p95_itl_ms', 'p99_itl_ms', 'total_requests', 'failed_requests', 
            'total_input_tokens', 'total_output_tokens', 'mean_ttft_ms', 'std_ttft_ms', 
            'mean_tpot_ms', 'std_tpot_ms', 'mean_itl_ms', 'std_itl_ms'
        ]
    
    @lru_cache(maxsize=32)
    def _get_request_fieldnames(self) -> List[str]:
        """Get fieldnames for request CSV (cached for performance)"""
        return [
            'run_id', 'model', 'suite', 'timestamp', 'request_id', 'input_tokens', 
            'output_tokens', 'ttft_ms', 'itl_ms', 'generated_text_length', 'error',
            'request_throughput', 'output_throughput'
        ]
    
    @lru_cache(maxsize=32)
    def _get_system_fieldnames(self) -> List[str]:
        """Get fieldnames for system CSV (cached for performance)"""
        return [
            'run_id', 'model', 'suite', 'timestamp', 'cpu_percent', 'cpu_count', 
            'cpu_count_logical', 'memory_total_gb', 'memory_available_gb', 'memory_used_gb', 
            'memory_percent', 'load_avg_1min', 'load_avg_5min', 'load_avg_15min',
            'peak_cpu_percent', 'avg_cpu_percent', 'peak_memory_gb', 'avg_memory_gb',
            'peak_gpu_memory_gb', 'avg_gpu_utilization'
        ]
