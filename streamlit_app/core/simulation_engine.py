"""Simulation engine wrapper for managing FGEM simulations."""

import sys
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING
from .config_manager import ConfigManager

if TYPE_CHECKING:
    from ..fgem.world import World

# Lazy import to avoid dependency issues during testing
def _import_world():
    from ..fgem.world import World
    return World


class SimulationEngine:
    """Wrapper class for managing simulation execution and progress tracking."""
    
    def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
        """Initialize simulation engine.
        
        Args:
            config: Configuration dictionary.
            config_manager: Optional ConfigManager instance.
        """
        self.config = config
        self.config_manager = config_manager or ConfigManager()
        self.world: Optional[World] = None
        self.state: str = "idle"  # idle, running, paused, completed, error
        self.progress: float = 0.0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.error_message: Optional[str] = None
        self.progress_callback: Optional[Callable[[float, int, int], None]] = None
        self.initialized_config: Optional[Dict[str, Any]] = None  # Store the config used for initialization
    
    def initialize(self, reset_market_weather: bool = True) -> bool:
        """Initialize the World instance.
        
        Args:
            reset_market_weather: Whether to reset market and weather data.
            
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Merge config with defaults
            # If config has a _source_file attribute, use it for path resolution
            config_file_path = getattr(self.config, '_source_file', None) if hasattr(self.config, '_source_file') else None
            merged_config = self.config_manager.merge_config(self.config, config_file_path=config_file_path)
            
            # Validate config
            is_valid, errors = self.config_manager.validate_config(merged_config)
            if not is_valid:
                self.error_message = "; ".join(errors)
                self.state = "error"
                return False
            
            # Create World instance (lazy import)
            World = _import_world()
            self.world = World(merged_config, 
                             reset_market_weather=reset_market_weather,
                             config_manager=self.config_manager)
            
            # Store the merged config that was used for initialization
            self.initialized_config = merged_config.copy()
            self.total_steps = self.world.max_simulation_steps
            self.state = "idle"
            return True
            
        except Exception as e:
            import traceback
            self.error_message = f"{str(e)}\n{traceback.format_exc()}"
            self.state = "error"
            return False
    
    def run(self, progress_callback: Optional[Callable[[float, int, int], None]] = None) -> bool:
        """Run the simulation.
        
        Args:
            progress_callback: Optional callback function(progress, current_step, total_steps).
            
        Returns:
            True if simulation completed successfully, False otherwise.
        """
        if self.world is None:
            if not self.initialize():
                return False
        
        self.progress_callback = progress_callback
        self.state = "running"
        self.current_step = 0
        
        try:
            for i in range(self.world.max_simulation_steps):
                self.world.step_update_record()
                self.current_step = i + 1
                self.progress = (self.current_step / self.total_steps) * 100
                
                # Call progress callback if provided
                if self.progress_callback:
                    self.progress_callback(self.progress, self.current_step, self.total_steps)
            
            # Post-process results
            self.world.postprocess(print_outputs=False, compute_pumping=True)
            
            self.state = "completed"
            return True
            
        except Exception as e:
            import traceback
            self.error_message = f"{str(e)}\n{traceback.format_exc()}"
            self.state = "error"
            return False
    
    def pause(self):
        """Pause the simulation (not implemented for now)."""
        if self.state == "running":
            self.state = "paused"
    
    def resume(self):
        """Resume the simulation (not implemented for now)."""
        if self.state == "paused":
            self.state = "running"
    
    def get_results(self) -> Optional['World']:
        """Get simulation results.
        
        Returns:
            World instance if simulation completed, None otherwise.
        """
        if self.state == "completed" and self.world is not None:
            return self.world
        return None
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary with progress information.
        """
        return {
            "state": self.state,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "error_message": self.error_message
        }
    
    def reset(self):
        """Reset the simulation engine."""
        self.world = None
        self.state = "idle"
        self.progress = 0.0
        self.current_step = 0
        self.total_steps = 0
        self.error_message = None
        self.progress_callback = None
        self.initialized_config = None

