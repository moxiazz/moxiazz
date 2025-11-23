"""
Visualization utilities for chaotic sequences and diffusion processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_sequence(
    sequence: np.ndarray,
    title: str = "Chaotic Sequence",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None
):
    """
    Plot a chaotic sequence.
    
    Args:
        sequence: The sequence to plot (1D or 2D array)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=figsize)
    
    if len(sequence.shape) == 1:
        # 1D sequence
        plt.plot(sequence, linewidth=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    else:
        # Multi-dimensional sequence
        n_dims = sequence.shape[1]
        if n_dims <= 3:
            for i in range(n_dims):
                plt.plot(sequence[:, i], label=f'Dim {i+1}', linewidth=0.5, alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # For higher dimensions, plot heatmap
            plt.imshow(sequence.T, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.xlabel('Time Step')
            plt.ylabel('Dimension')
            plt.title(title)
            plt.colorbar(label='Value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_attractor(
    sequence: np.ndarray,
    title: str = "Chaotic Attractor",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot the attractor of a chaotic sequence.
    
    Args:
        sequence: The sequence to plot (must be 2D or 3D)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    if len(sequence.shape) == 1:
        raise ValueError("Cannot plot attractor for 1D sequence")
    
    n_dims = sequence.shape[1]
    
    if n_dims == 2:
        # 2D attractor
        plt.figure(figsize=figsize)
        plt.scatter(sequence[:, 0], sequence[:, 1], s=1, alpha=0.5, c=range(len(sequence)), cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.colorbar(label='Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    elif n_dims == 3:
        # 3D attractor
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            sequence[:, 0], sequence[:, 1], sequence[:, 2],
            s=1, alpha=0.5, c=range(len(sequence)), cmap='viridis'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.colorbar(scatter, label='Time')
    else:
        # For higher dimensions, plot pairwise
        fig, axes = plt.subplots(min(n_dims, 3), min(n_dims, 3), figsize=figsize)
        for i in range(min(n_dims, 3)):
            for j in range(min(n_dims, 3)):
                if i == j:
                    axes[i, j].hist(sequence[:, i], bins=50, alpha=0.7)
                    axes[i, j].set_ylabel('Frequency')
                else:
                    axes[i, j].scatter(sequence[:, j], sequence[:, i], s=1, alpha=0.3)
                    axes[i, j].set_xlabel(f'Dim {j+1}')
                    axes[i, j].set_ylabel(f'Dim {i+1}')
        plt.suptitle(title)
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_diffusion_process(
    sequences: List[np.ndarray],
    timesteps: List[int],
    title: str = "Diffusion Process",
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
):
    """
    Visualize the diffusion process at different timesteps.
    
    Args:
        sequences: List of sequences at different timesteps
        timesteps: Corresponding timesteps
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    n_steps = len(sequences)
    
    if n_steps == 0:
        return
    
    # Determine layout
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, (seq, t) in enumerate(zip(sequences, timesteps)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if len(seq.shape) == 1:
            ax.plot(seq, linewidth=0.5)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
        else:
            # For multi-dimensional, plot first dimension or show heatmap
            if seq.shape[1] <= 3:
                for i in range(seq.shape[1]):
                    ax.plot(seq[:, i], linewidth=0.5, alpha=0.7, label=f'Dim {i+1}')
                ax.legend(fontsize=8)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
            else:
                im = ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Dimension')
                plt.colorbar(im, ax=ax)
        
        ax.set_title(f't = {t}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_steps, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_sequences(
    original: np.ndarray,
    generated: np.ndarray,
    title: str = "Original vs Generated",
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
):
    """
    Compare original and generated sequences.
    
    Args:
        original: Original sequence
        generated: Generated sequence
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original
    if len(original.shape) == 1:
        axes[0].plot(original, linewidth=0.5, color='blue')
        axes[1].plot(generated, linewidth=0.5, color='red')
    else:
        for i in range(min(original.shape[1], 3)):
            axes[0].plot(original[:, i], linewidth=0.5, alpha=0.7, label=f'Dim {i+1}')
            axes[1].plot(generated[:, i], linewidth=0.5, alpha=0.7, label=f'Dim {i+1}')
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    
    axes[0].set_title('Original Sequence')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Generated Sequence')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    # Plot difference
    diff = np.abs(original - generated)
    if len(diff.shape) == 1:
        axes[2].plot(diff, linewidth=0.5, color='green')
    else:
        for i in range(min(diff.shape[1], 3)):
            axes[2].plot(diff[:, i], linewidth=0.5, alpha=0.7, label=f'Dim {i+1}')
        axes[2].legend(fontsize=8)
    
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('|Difference|')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
