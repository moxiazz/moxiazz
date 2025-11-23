"""
Chaotic Systems Implementation
This module provides various chaotic systems that can be used to generate chaotic sequences.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class ChaoticSystem(ABC):
    """Base class for chaotic systems."""
    
    @abstractmethod
    def step(self, state: np.ndarray) -> np.ndarray:
        """Perform one iteration of the chaotic system."""
        pass
    
    @abstractmethod
    def generate_sequence(self, length: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a chaotic sequence of specified length."""
        pass


class LogisticMap(ChaoticSystem):
    """
    Logistic Map: x_{n+1} = r * x_n * (1 - x_n)
    A simple 1D chaotic map that exhibits chaotic behavior for certain parameter values.
    """
    
    def __init__(self, r: float = 3.9):
        """
        Initialize the Logistic Map.
        
        Args:
            r: Control parameter. Chaotic behavior typically occurs for r in [3.57, 4.0]
        """
        self.r = r
    
    def step(self, state: np.ndarray) -> np.ndarray:
        """Perform one iteration of the logistic map."""
        return self.r * state * (1 - state)
    
    def generate_sequence(self, length: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a chaotic sequence using the logistic map.
        
        Args:
            length: Length of the sequence to generate
            initial_state: Initial value (default: random value in [0, 1])
            
        Returns:
            Array of shape (length,) containing the chaotic sequence
        """
        if initial_state is None:
            initial_state = np.random.rand()
        
        sequence = np.zeros(length)
        sequence[0] = initial_state
        
        for i in range(1, length):
            sequence[i] = self.step(sequence[i-1])
        
        return sequence


class HenonMap(ChaoticSystem):
    """
    Henon Map: 
    x_{n+1} = 1 - a * x_n^2 + y_n
    y_{n+1} = b * x_n
    
    A 2D chaotic map with complex attractor structure.
    """
    
    def __init__(self, a: float = 1.4, b: float = 0.3):
        """
        Initialize the Henon Map.
        
        Args:
            a: First control parameter (default: 1.4)
            b: Second control parameter (default: 0.3)
        """
        self.a = a
        self.b = b
    
    def step(self, state: np.ndarray) -> np.ndarray:
        """Perform one iteration of the Henon map."""
        x, y = state
        x_new = 1 - self.a * x**2 + y
        y_new = self.b * x
        return np.array([x_new, y_new])
    
    def generate_sequence(self, length: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a chaotic sequence using the Henon map.
        
        Args:
            length: Length of the sequence to generate
            initial_state: Initial 2D state (default: random values)
            
        Returns:
            Array of shape (length, 2) containing the chaotic sequence
        """
        if initial_state is None:
            initial_state = np.random.randn(2) * 0.1
        
        sequence = np.zeros((length, 2))
        sequence[0] = initial_state
        
        for i in range(1, length):
            sequence[i] = self.step(sequence[i-1])
        
        return sequence


class LorenzSystem(ChaoticSystem):
    """
    Lorenz System (discrete approximation):
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    
    A 3D chaotic system with butterfly attractor.
    """
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3, dt: float = 0.01):
        """
        Initialize the Lorenz System.
        
        Args:
            sigma: Prandtl number (default: 10.0)
            rho: Rayleigh number (default: 28.0)
            beta: Geometric parameter (default: 8/3)
            dt: Time step for numerical integration (default: 0.01)
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
    
    def step(self, state: np.ndarray) -> np.ndarray:
        """Perform one iteration of the Lorenz system using Euler method."""
        x, y, z = state
        
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        x_new = x + self.dt * dx
        y_new = y + self.dt * dy
        z_new = z + self.dt * dz
        
        return np.array([x_new, y_new, z_new])
    
    def generate_sequence(self, length: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a chaotic sequence using the Lorenz system.
        
        Args:
            length: Length of the sequence to generate
            initial_state: Initial 3D state (default: random values near origin)
            
        Returns:
            Array of shape (length, 3) containing the chaotic sequence
        """
        if initial_state is None:
            initial_state = np.random.randn(3)
        
        sequence = np.zeros((length, 3))
        sequence[0] = initial_state
        
        for i in range(1, length):
            sequence[i] = self.step(sequence[i-1])
        
        return sequence
