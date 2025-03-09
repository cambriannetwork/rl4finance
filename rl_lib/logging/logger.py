"""
Logger implementation for RL library.

This module provides JSON-formatted logging functionality for the RL library.
Logs are written to timestamped files in a 'logs' directory.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, Union, List
import numpy as np

# Create a custom JSON formatter that can handle numpy arrays and other complex types
class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for logging."""
    
    def __init__(self):
        super().__init__()
    
    def _serialize(self, obj: Any) -> Any:
        """Serialize objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists with limited size
            if obj.size > 100:  # Only show a sample for large arrays
                shape_str = 'x'.join(str(dim) for dim in obj.shape)
                sample = obj.flatten()[:5].tolist()
                return f"ndarray({shape_str}): sample={sample}..."
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            # Handle lists and tuples recursively
            if len(obj) > 100:  # Only show a sample for large lists
                return [self._serialize(item) for item in list(obj)[:5]] + ["..."]
            return [self._serialize(item) for item in obj]
        elif isinstance(obj, dict):
            # Handle dictionaries recursively
            return {k: self._serialize(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # For custom objects, convert to dict
            return {
                "__type": obj.__class__.__name__,
                **{k: self._serialize(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
            }
        return obj
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Handle the case where the message is already a dict
        if isinstance(record.msg, dict):
            log_data['data'] = self._serialize(record.msg)
        else:
            log_data['message'] = record.msg
            
            # Add any extra attributes
            if hasattr(record, 'data'):
                log_data['data'] = self._serialize(record.data)
        
        return json.dumps(log_data)


# Global logger instance
_logger = None

def setup_logger(
    debug: bool = False,
    log_level: str = "info",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up the logger with the specified configuration.
    
    Args:
        debug: Whether to enable debugging
        log_level: The log level (debug, info, warning, error)
        log_file: Optional custom log file path
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create logger
    logger = logging.getLogger("rl_lib")
    
    # Set log level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    
    # Set the level based on debug flag and log_level
    if debug:
        logger.setLevel(level_map.get(log_level.lower(), logging.INFO))
    else:
        logger.setLevel(logging.WARNING)  # Minimal logging when debug is False
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file if not specified
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"rl_lib_{timestamp}.json")
    elif not os.path.isabs(log_file):
        # If relative path, put it in the logs directory
        log_file = os.path.join(logs_dir, log_file)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JsonFormatter())
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Store the logger
    _logger = logger
    
    # Log initial setup
    if debug:
        logger.info({
            "event": "logger_initialized",
            "log_level": log_level,
            "log_file": log_file
        })
    
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.
    
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        # Set up with default configuration if not already configured
        _logger = setup_logger()
    
    return _logger


# Helper functions for common logging patterns

def log_iteration(
    iteration: int,
    total: int,
    metrics: Dict[str, Any],
    log_frequency: int = 10
) -> None:
    """
    Log iteration progress with metrics at specified frequency.
    
    Args:
        iteration: Current iteration number
        total: Total number of iterations
        metrics: Dictionary of metrics to log
        log_frequency: How often to log (e.g., every 10 iterations)
    """
    logger = get_logger()
    
    # Only log at specified frequency or at start/end
    if (iteration % log_frequency == 0 or 
        iteration == 0 or 
        iteration == total - 1):
        
        logger.info({
            "event": "iteration_progress",
            "iteration": iteration,
            "total": total,
            "progress": f"{iteration}/{total} ({iteration/total:.1%})",
            "metrics": metrics
        })


def log_weights_summary(weights, name: str = "weights") -> None:
    """
    Log a summary of weights (mean, min, max, etc.).
    
    Args:
        weights: Weights to summarize (numpy array or list of arrays)
        name: Name to identify these weights in the log
    """
    logger = get_logger()
    
    if isinstance(weights, list):
        # Handle list of weight arrays (e.g., for multiple layers)
        summaries = []
        for i, w in enumerate(weights):
            if hasattr(w, 'weights'):
                # Handle Weights objects
                w_array = w.weights
            else:
                w_array = w
                
            summaries.append({
                "layer": i,
                "shape": w_array.shape,
                "mean": float(np.mean(w_array)),
                "std": float(np.std(w_array)),
                "min": float(np.min(w_array)),
                "max": float(np.max(w_array)),
                "sample": w_array.flatten()[:3].tolist()
            })
        
        logger.debug({
            "event": f"{name}_summary",
            "layers": len(summaries),
            "summaries": summaries
        })
    else:
        # Handle single weight array
        if hasattr(weights, 'weights'):
            # Handle Weights objects
            w_array = weights.weights
        else:
            w_array = weights
            
        logger.debug({
            "event": f"{name}_summary",
            "shape": w_array.shape,
            "mean": float(np.mean(w_array)),
            "std": float(np.std(w_array)),
            "min": float(np.min(w_array)),
            "max": float(np.max(w_array)),
            "sample": w_array.flatten()[:5].tolist()
        })


def log_phase(phase: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the start of a new processing phase.
    
    Args:
        phase: Name of the phase
        details: Optional details about the phase
    """
    logger = get_logger()
    
    log_data = {
        "event": "phase_start",
        "phase": phase
    }
    
    if details:
        log_data["details"] = details
        
    logger.info(log_data)


def log_convergence(
    iteration: int, 
    error: float, 
    tolerance: float,
    converged: bool
) -> None:
    """
    Log convergence information.
    
    Args:
        iteration: Current iteration
        error: Current error
        tolerance: Error tolerance for convergence
        converged: Whether convergence has been achieved
    """
    logger = get_logger()
    
    logger.info({
        "event": "convergence_check",
        "iteration": iteration,
        "error": error,
        "tolerance": tolerance,
        "converged": converged
    })


def log_mdp_step(
    state: Any,
    action: Any,
    next_state: Any,
    reward: float,
    log_frequency: int = 100,
    step_count: int = 0
) -> None:
    """
    Log MDP state transitions at specified frequency.
    
    Args:
        state: Current state
        action: Action taken
        next_state: Resulting state
        reward: Reward received
        log_frequency: How often to log
        step_count: Current step count (for frequency calculation)
    """
    logger = get_logger()
    
    # Only log at specified frequency
    if step_count % log_frequency == 0:
        logger.debug({
            "event": "mdp_step",
            "step": step_count,
            "state": str(state),
            "action": action,
            "next_state": str(next_state),
            "reward": reward
        })
