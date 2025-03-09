"""
State classes for Markov processes.

This module provides classes for representing states in Markov processes,
including terminal and non-terminal states.
"""

from typing import TypeVar, Generic, Callable, Any

# Type variable for state values
S = TypeVar('S')

class State(Generic[S]):
    """
    Base class for states in a Markov process.
    
    This class represents a state in a Markov process. States can be either
    terminal or non-terminal, and they contain a value of type S.
    """
    
    def __init__(self, state_value: S):
        """
        Initialize a state with a value.
        
        Args:
            state_value: The value associated with this state
        """
        self.state = state_value
    
    def on_non_terminal(self, f: Callable[['NonTerminal[S]'], Any], default: Any) -> Any:
        """
        Apply function f if this is a non-terminal state, otherwise return default.
        
        This method provides a convenient way to handle terminal and non-terminal
        states differently without explicit type checking.
        
        Args:
            f: Function to apply if this is a non-terminal state
            default: Value to return if this is a terminal state
            
        Returns:
            Result of f(self) if non-terminal, default otherwise
        """
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default
    
    def __repr__(self) -> str:
        """
        Return a string representation of the state.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}({self.state})"


class Terminal(State[S]):
    """
    Terminal state in a Markov process.
    
    A terminal state is a state from which no further transitions are possible.
    It represents the end of an episode or process.
    """
    
    def __init__(self, state_value: S):
        """
        Initialize a terminal state with a value.
        
        Args:
            state_value: The value associated with this terminal state
        """
        super().__init__(state_value)


class NonTerminal(State[S]):
    """
    Non-terminal state in a Markov process.
    
    A non-terminal state is a state from which further transitions are possible.
    It represents an intermediate point in an episode or process.
    """
    
    def __init__(self, state_value: S):
        """
        Initialize a non-terminal state with a value.
        
        Args:
            state_value: The value associated with this non-terminal state
        """
        super().__init__(state_value)
    
    def __eq__(self, other) -> bool:
        """
        Check if this state is equal to another state.
        
        Two non-terminal states are equal if their state values are equal.
        
        Args:
            other: Another state to compare with
            
        Returns:
            True if the states are equal, False otherwise
        """
        if not isinstance(other, NonTerminal):
            return False
        return self.state == other.state
    
    def __lt__(self, other) -> bool:
        """
        Check if this state is less than another state.
        
        This method enables sorting of non-terminal states.
        
        Args:
            other: Another state to compare with
            
        Returns:
            True if this state is less than the other state, False otherwise
        """
        if not isinstance(other, NonTerminal):
            raise TypeError("Can only compare NonTerminal states")
        return self.state < other.state
    
    def __hash__(self) -> int:
        """
        Compute a hash value for this state.
        
        Returns:
            Hash value
        """
        return hash((NonTerminal, self.state))
