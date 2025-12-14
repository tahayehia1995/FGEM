"""Test all example configurations and measure execution times."""

import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.core.config_manager import ConfigManager
from streamlit_app.core.simulation_engine import SimulationEngine

def test_example(example_path: str) -> tuple[bool, float, str]:
    """Test a single example configuration.
    
    Args:
        example_path: Path to example JSON file
        
    Returns:
        Tuple of (success: bool, execution_time: float, error_message: str)
    """
    try:
        # Load and merge config
        config_manager = ConfigManager()
        with open(example_path, 'r') as f:
            example_config = json.load(f)
        
        config = config_manager.merge_config(example_config, config_file_path=example_path)
        
        # Validate config
        is_valid, errors = config_manager.validate_config(config)
        if not is_valid:
            return False, 0.0, f"Validation errors: {', '.join(errors)}"
        
        # Run simulation
        start_time = time.time()
        engine = SimulationEngine(config)
        engine.run()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Check if simulation completed successfully
        results = engine.get_results()
        if results is None:
            return False, execution_time, "Simulation completed but no results returned"
        
        return True, execution_time, ""
        
    except Exception as e:
        return False, 0.0, str(e)

def main():
    """Test all examples and report execution times."""
    examples_dir = Path("examples/configs")
    
    # Get all example files
    example_files = sorted(examples_dir.glob("example*.json"))
    
    print("=" * 80)
    print("Testing All Example Configurations")
    print("=" * 80)
    print()
    
    results = []
    
    for example_file in example_files:
        print(f"Testing {example_file.name}...", end=" ", flush=True)
        
        success, exec_time, error = test_example(str(example_file))
        
        if success:
            status = "✓ SUCCESS"
            if exec_time > 180:  # > 3 minutes
                status += " ⚠️ SLOW (>3 min)"
            print(f"{status} - {exec_time:.2f} seconds ({exec_time/60:.2f} minutes)")
        else:
            print(f"✗ FAILED - {error}")
            exec_time = None
        
        results.append({
            "file": example_file.name,
            "success": success,
            "time_seconds": exec_time,
            "time_minutes": exec_time / 60 if exec_time else None,
            "error": error
        })
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    # Sort by execution time (slowest first)
    successful_results = [r for r in results if r["success"]]
    successful_results.sort(key=lambda x: x["time_seconds"] if x["time_seconds"] else 0, reverse=True)
    
    print("Execution Times (slowest first):")
    print("-" * 80)
    for r in successful_results:
        time_str = f"{r['time_seconds']:.2f}s ({r['time_minutes']:.2f} min)" if r['time_seconds'] else "N/A"
        print(f"  {r['file']:40s} {time_str}")
    
    print()
    print("Examples taking > 3 minutes:")
    print("-" * 80)
    slow_examples = [r for r in successful_results if r["time_seconds"] and r["time_seconds"] > 180]
    if slow_examples:
        for r in slow_examples:
            print(f"  {r['file']:40s} {r['time_seconds']:.2f}s ({r['time_minutes']:.2f} min)")
    else:
        print("  None")
    
    print()
    print("Failed Examples:")
    print("-" * 80)
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        for r in failed_results:
            print(f"  {r['file']:40s} {r['error']}")
    else:
        print("  None")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
