"""
Advanced example demonstrating multiple chaotic systems and degradation operators.

This example shows:
1. Different chaotic systems (Logistic, Henon, Lorenz)
2. Different degradation operators
3. Comparison of results
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chaotic_diffusion.core import LogisticMap, HenonMap, LorenzSystem
from chaotic_diffusion.models import ChaoticColdDiffusion, ChaoticDegradation
from chaotic_diffusion.models.degradation import PermutationDegradation, MaskingDegradation


def test_chaotic_system(system_name, system, sequence_length, sequence_dim):
    """Test a chaotic system with Cold Diffusion."""
    print(f"\n{'='*60}")
    print(f"Testing: {system_name}")
    print(f"{'='*60}")
    
    # Generate sequence
    sequence = system.generate_sequence(length=sequence_length)
    print(f"Generated sequence shape: {sequence.shape}")
    if len(sequence.shape) == 1:
        print(f"Value range: [{sequence.min():.4f}, {sequence.max():.4f}]")
    else:
        for dim in range(sequence.shape[1]):
            print(f"Dim {dim+1} range: [{sequence[:, dim].min():.4f}, {sequence[:, dim].max():.4f}]")
    
    return sequence


def test_degradation_operator(degradation_name, degradation, sequence, T=50):
    """Test a degradation operator."""
    print(f"\n  Testing degradation: {degradation_name}")
    
    # Test forward process
    degraded = {}
    for t in [0, T//4, T//2, 3*T//4, T]:
        if hasattr(degradation, 'degrade'):
            x_t = degradation.degrade(sequence, t, T)
        else:
            x_t = sequence.copy()  # Fallback
        
        if len(x_t.shape) == 1:
            print(f"    t={t:3d}: Mean={x_t.mean():7.4f}, Std={x_t.std():7.4f}")
        else:
            print(f"    t={t:3d}: Mean={x_t.mean():7.4f}, Std={x_t.std():7.4f}, Shape={x_t.shape}")
        
        degraded[t] = x_t
    
    return degraded


def main():
    """Run advanced example."""
    print("=" * 60)
    print("Chaotic Cold Diffusion - Advanced Example")
    print("=" * 60)
    
    # Test different chaotic systems
    print("\n📌 Part 1: Testing Different Chaotic Systems")
    
    systems = {
        "Logistic Map": (LogisticMap(r=3.9), 200, 1),
        "Henon Map": (HenonMap(a=1.4, b=0.3), 200, 2),
        "Lorenz System": (LorenzSystem(sigma=10.0, rho=28.0, beta=8/3), 200, 3),
    }
    
    sequences = {}
    for name, (system, length, dim) in systems.items():
        sequences[name] = test_chaotic_system(name, system, length, dim)
    
    # Test different degradation operators
    print("\n\n📌 Part 2: Testing Different Degradation Operators")
    
    degradations = {
        "Chaotic (Linear)": ChaoticDegradation(noise_schedule="linear"),
        "Chaotic (Cosine)": ChaoticDegradation(noise_schedule="cosine"),
        "Permutation": PermutationDegradation(seed=42),
        "Masking": MaskingDegradation(mask_value=0.0, seed=42),
    }
    
    # Test each degradation on Logistic Map sequence
    print("\nApplying degradations to Logistic Map sequence:")
    test_sequence = sequences["Logistic Map"]
    
    for deg_name, degradation in degradations.items():
        test_degradation_operator(deg_name, degradation, test_sequence)
    
    # Test Cold Diffusion with different configurations
    print("\n\n📌 Part 3: Cold Diffusion with Different Configurations")
    
    configs = [
        ("Logistic + Chaotic Degradation", sequences["Logistic Map"], ChaoticDegradation(), 1),
        ("Henon + Permutation", sequences["Henon Map"], PermutationDegradation(seed=42), 2),
        ("Lorenz + Masking", sequences["Lorenz System"], MaskingDegradation(seed=42), 3),
    ]
    
    for config_name, seq, deg, dim in configs:
        print(f"\n{config_name}:")
        
        diffusion = ChaoticColdDiffusion(
            degradation=deg,
            T=50,
            sequence_length=len(seq),
            sequence_dim=dim
        )
        
        # Test forward process
        x_25 = diffusion.forward_process(seq, 25)
        x_50 = diffusion.forward_process(seq, 50)
        
        print(f"  Forward process t=25: Mean={x_25.mean():.4f}")
        print(f"  Forward process t=50: Mean={x_50.mean():.4f}")
        
        # Test loss computation
        loss = diffusion.compute_loss(seq, n_steps=5)
        print(f"  Average loss: {loss:.6f}")
    
    print("\n" + "=" * 60)
    print("Advanced example completed successfully!")
    print("=" * 60)
    
    # Summary
    print("\n📊 Summary:")
    print(f"  Tested {len(systems)} chaotic systems")
    print(f"  Tested {len(degradations)} degradation operators")
    print(f"  Tested {len(configs)} Cold Diffusion configurations")


if __name__ == "__main__":
    main()
