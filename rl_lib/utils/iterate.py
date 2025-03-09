"""
Iteration utilities for RL algorithms.

This module provides utility functions for iterative algorithms,
including functions for convergence detection and accumulation.
"""

import itertools
from typing import TypeVar, Callable, Iterator, Optional, Iterable, Any

# Type variable for values
T = TypeVar('T')
U = TypeVar('U')

def iterate(step_func: Callable[[T], T], start_value: T) -> Iterator[T]:
    """
    Generate a sequence by repeatedly applying a function to its own result.
    
    This function creates an iterator that yields:
    start_value, step_func(start_value), step_func(step_func(start_value)), ...
    
    Args:
        step_func: Function to apply repeatedly
        start_value: Initial value
        
    Returns:
        Iterator yielding values in sequence
    """
    state = start_value
    while True:
        yield state
        state = step_func(state)


def converge(values_iterator: Iterator[T], done_func: Callable[[T, T], bool]) -> Iterator[T]:
    """
    Read from an iterator until two consecutive values satisfy the done function.
    
    This function yields values from the input iterator until the done function
    returns True for two consecutive values, or the input iterator is exhausted.
    
    Args:
        values_iterator: Iterator of values
        done_func: Function that takes two consecutive values and returns True if converged
        
    Returns:
        Iterator that stops when convergence is detected
    """
    # Get the first value
    try:
        a = next(values_iterator)
    except StopIteration:
        # Empty iterator, nothing to converge
        return
    
    # Yield the first value
    yield a
    
    # Process remaining values
    for b in values_iterator:
        yield b
        
        # Check for convergence
        if done_func(a, b):
            return
        
        # Update a for next iteration
        a = b


def converged(
    values_iterator: Iterator[T],
    done_func: Optional[Callable[[T, T], bool]] = None,
    *,
    done: Optional[Callable[[T, T], bool]] = None
) -> T:
    """
    Return the final value when an iterator converges according to the done function.
    
    This function consumes the input iterator until convergence is detected,
    then returns the final value.
    
    Args:
        values_iterator: Iterator of values
        done_func: Function that takes two consecutive values and returns True if converged
        done: Alternative name for done_func (for backward compatibility)
        
    Returns:
        The final value after convergence
        
    Raises:
        ValueError: If done_func and done are both None, or if the iterator is empty
    """
    # Handle both done_func and done parameters for compatibility
    if done is not None:
        done_func = done
    
    if done_func is None:
        raise ValueError("Either done_func or done must be provided")
    
    # Initialize result to None
    result = None
    
    # Iterate until convergence
    for x in converge(values_iterator, done_func):
        result = x
    
    # Check if we got any values
    if result is None:
        raise ValueError("converged called on an empty iterator")
    
    return result


def accumulate(
    iterable: Iterable[T],
    func: Callable[[U, T], U],
    *,
    initial: Optional[U] = None
) -> Iterator[U]:
    """
    Make an iterator that returns accumulated results of a binary function.
    
    This function is similar to itertools.accumulate but allows an initial value.
    
    Args:
        iterable: Input iterable
        func: Binary function to apply
        initial: Optional initial value
        
    Returns:
        Iterator of accumulated values
    """
    # If initial value is provided, prepend it to the iterable
    if initial is not None:
        iterable = itertools.chain([initial], iterable)
    
    # Use itertools.accumulate for the actual accumulation
    return itertools.accumulate(iterable, func)
