"""
Degradation Operators for Chaotic Sequences
This module implements various degradation operators that can be used in Cold Diffusion.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

# Numerical stability constant
EPSILON = 1e-8


class DegradationOperator(ABC):
    """Base class for degradation operators."""
    
    @abstractmethod
    def degrade(self, x: np.ndarray, t: int, T: int) -> np.ndarray:
        """
        Apply degradation at timestep t.
        
        Args:
            x: Input sequence
            t: Current timestep
            T: Total number of timesteps
            
        Returns:
            Degraded sequence
        """
        pass
    
    @abstractmethod
    def restore(self, x_t: np.ndarray, x_pred: np.ndarray, t: int, T: int) -> np.ndarray:
        """
        Restore one step from degraded sequence.
        
        Args:
            x_t: Degraded sequence at timestep t
            x_pred: Predicted restoration
            t: Current timestep
            T: Total number of timesteps
            
        Returns:
            Restored sequence at timestep t-1
        """
        pass


class ChaoticDegradation(DegradationOperator):
    """
    Chaotic-aware degradation operator.
    
    This operator gradually destroys the chaotic structure by:
    1. Adding noise that increases with timestep
    2. Mixing with random sequences
    3. Smoothing that destroys fine-scale chaotic structure
    """
    
    def __init__(self, noise_schedule: str = "linear", beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize chaotic degradation operator.
        
        Args:
            noise_schedule: Type of noise schedule ("linear", "cosine")
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.noise_schedule = noise_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_beta(self, t: int, T: int) -> float:
        """Get noise level at timestep t."""
        if self.noise_schedule == "linear":
            return self.beta_start + (self.beta_end - self.beta_start) * (t / T)
        elif self.noise_schedule == "cosine":
            # Small offset to prevent boundary issues (from DDPM paper)
            s = 0.008
            f_t = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
            f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
            return np.clip(1 - f_t / f_0, 0, 0.999)
        else:
            return self.beta_start + (self.beta_end - self.beta_start) * (t / T)
    
    def degrade(self, x: np.ndarray, t: int, T: int) -> np.ndarray:
        """
        Apply progressive degradation to chaotic sequence.
        
        The degradation process gradually destroys chaotic structure:
        - At t=0: original sequence
        - At t=T: heavily degraded/random sequence
        """
        if t == 0:
            return x.copy()
        
        beta = self.get_beta(t, T)
        alpha = 1 - beta
        
        # Progressive mixing with noise
        noise = np.random.randn(*x.shape)
        x_t = np.sqrt(alpha) * x + np.sqrt(beta) * noise
        
        # Additional smoothing for higher t values
        if t > T // 2:
            smooth_factor = (t - T // 2) / (T // 2)
            kernel_size = int(3 + smooth_factor * 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Simple moving average smoothing
            if len(x.shape) == 1:
                x_t = self._smooth_1d(x_t, kernel_size)
            else:
                for i in range(x.shape[1]):
                    x_t[:, i] = self._smooth_1d(x_t[:, i], kernel_size)
        
        return x_t
    
    def _smooth_1d(self, x: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply simple moving average smoothing."""
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(x, kernel, mode='same')
    
    def restore(self, x_t: np.ndarray, x_pred: np.ndarray, t: int, T: int) -> np.ndarray:
        """
        Restore one step using predicted sequence.
        
        This is the reverse of the degradation process.
        """
        if t == 0:
            return x_pred
        
        beta = self.get_beta(t, T)
        alpha = 1 - beta
        
        # Estimate noise that was added
        noise_est = (x_t - np.sqrt(alpha) * x_pred) / np.sqrt(beta + EPSILON)
        
        # Remove one step of degradation
        beta_prev = self.get_beta(t - 1, T) if t > 1 else 0
        alpha_prev = 1 - beta_prev
        
        x_prev = (x_t - np.sqrt(beta) * noise_est) / np.sqrt(alpha + EPSILON)
        
        # Add smaller noise for previous step
        if t > 1:
            z = np.random.randn(*x_t.shape)
            x_prev = x_prev + np.sqrt(beta_prev) * z
        
        return x_prev


class PermutationDegradation(DegradationOperator):
    """
    Degradation by gradually permuting/shuffling the sequence.
    
    This destroys temporal correlations in the chaotic sequence.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize permutation degradation.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def degrade(self, x: np.ndarray, t: int, T: int) -> np.ndarray:
        """Apply progressive permutation degradation."""
        if t == 0:
            return x.copy()
        
        # Fraction of elements to permute
        permute_fraction = t / T
        n_permute = int(len(x) * permute_fraction)
        
        x_t = x.copy()
        if n_permute > 0:
            indices = self.rng.choice(len(x), size=n_permute, replace=False)
            permuted_values = x_t[indices].copy()
            self.rng.shuffle(permuted_values)
            x_t[indices] = permuted_values
        
        return x_t
    
    def restore(self, x_t: np.ndarray, x_pred: np.ndarray, t: int, T: int) -> np.ndarray:
        """Restore by using predicted sequence."""
        # For permutation, restoration is challenging
        # We rely on the model's prediction
        return x_pred


class MaskingDegradation(DegradationOperator):
    """
    Degradation by gradually masking/removing elements.
    
    Similar to masked language modeling but for sequences.
    """
    
    def __init__(self, mask_value: float = 0.0, seed: Optional[int] = None):
        """
        Initialize masking degradation.
        
        Args:
            mask_value: Value to use for masked elements
            seed: Random seed for reproducibility
        """
        self.mask_value = mask_value
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def degrade(self, x: np.ndarray, t: int, T: int) -> np.ndarray:
        """Apply progressive masking degradation."""
        if t == 0:
            return x.copy()
        
        # Fraction of elements to mask
        mask_fraction = t / T
        n_mask = int(len(x) * mask_fraction)
        
        x_t = x.copy()
        if n_mask > 0:
            indices = self.rng.choice(len(x), size=n_mask, replace=False)
            x_t[indices] = self.mask_value
        
        return x_t
    
    def restore(self, x_t: np.ndarray, x_pred: np.ndarray, t: int, T: int) -> np.ndarray:
        """Restore by filling in predicted values."""
        # Identify masked positions
        mask = (x_t == self.mask_value)
        
        x_prev = x_t.copy()
        x_prev[mask] = x_pred[mask]
        
        return x_prev
