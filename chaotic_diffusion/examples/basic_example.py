"""
Basic example demonstrating the Chaotic Cold Diffusion framework.

This example shows how to:
1. Generate a chaotic sequence (Logistic Map)
2. Apply the forward diffusion process
3. Visualize the degradation process
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chaotic_diffusion.core import LogisticMap
from chaotic_diffusion.models import ChaoticColdDiffusion, ChaoticDegradation


def main():
    """Run basic example."""
    print("=" * 60)
    print("Chaotic Cold Diffusion - Basic Example")
    print("=" * 60)
    
    # Step 1: Generate a chaotic sequence using Logistic Map
    print("\n1. Generating chaotic sequence using Logistic Map...")
    logistic = LogisticMap(r=3.9)  # Chaotic parameter
    sequence_length = 200
    chaotic_sequence = logistic.generate_sequence(
        length=sequence_length,
        initial_state=np.array([0.1])
    )
    print(f"   Generated sequence of length {len(chaotic_sequence)}")
    print(f"   Value range: [{chaotic_sequence.min():.4f}, {chaotic_sequence.max():.4f}]")
    print(f"   Mean: {chaotic_sequence.mean():.4f}, Std: {chaotic_sequence.std():.4f}")
    
    # Step 2: Initialize Cold Diffusion model
    print("\n2. Initializing Cold Diffusion model...")
    degradation = ChaoticDegradation(noise_schedule="linear", beta_start=0.0001, beta_end=0.02)
    diffusion = ChaoticColdDiffusion(
        degradation=degradation,
        T=100,  # Number of diffusion steps
        sequence_length=sequence_length,
        sequence_dim=1
    )
    print(f"   Model initialized with T={diffusion.T} timesteps")
    
    # Step 3: Apply forward diffusion at different timesteps
    print("\n3. Applying forward diffusion process...")
    timesteps_to_visualize = [0, 25, 50, 75, 100]
    degraded_sequences = []
    
    for t in timesteps_to_visualize:
        x_t = diffusion.forward_process(chaotic_sequence, t)
        degraded_sequences.append(x_t)
        print(f"   t={t:3d}: Mean={x_t.mean():7.4f}, Std={x_t.std():7.4f}")
    
    # Step 4: Demonstrate reverse sampling
    print("\n4. Testing reverse diffusion (sampling)...")
    # Start from noise
    x_T = np.random.randn(sequence_length)
    generated = diffusion.sample(x_T, return_intermediates=False)
    print(f"   Generated sequence: Mean={generated.mean():.4f}, Std={generated.std():.4f}")
    
    # Step 5: Compute training loss
    print("\n5. Computing training loss...")
    loss = diffusion.compute_loss(chaotic_sequence, n_steps=10)
    print(f"   Average loss over 10 timesteps: {loss:.6f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Optional: Save some statistics
    print("\n📊 Summary Statistics:")
    print(f"  Original sequence:")
    print(f"    - Range: [{chaotic_sequence.min():.4f}, {chaotic_sequence.max():.4f}]")
    print(f"    - Mean ± Std: {chaotic_sequence.mean():.4f} ± {chaotic_sequence.std():.4f}")
    print(f"  Generated sequence:")
    print(f"    - Range: [{generated.min():.4f}, {generated.max():.4f}]")
    print(f"    - Mean ± Std: {generated.mean():.4f} ± {generated.std():.4f}")
    print(f"  MSE between original and generated: {np.mean((chaotic_sequence - generated)**2):.6f}")


if __name__ == "__main__":
    main()
