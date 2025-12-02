#!/usr/bin/env python3
"""
Simple test for Gumbel deletion components without database dependencies.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only the Gumbel-specific functions
from InferenceGraph.differential_deletion import sample_gumbel


def test_gumbel_sampling():
    """Test Gumbel noise sampling."""
    print("=" * 70)
    print("TEST: Gumbel Noise Sampling")
    print("=" * 70)

    # Sample multiple times and check distribution
    num_samples = 10000
    samples = [sample_gumbel() for _ in range(num_samples)]

    mean = np.mean(samples)
    std = np.std(samples)

    print(f"\nSampled {num_samples} Gumbel(0,1) values")
    print(f"  Mean: {mean:.3f} (theoretical: 0.577)")
    print(f"  Std:  {std:.3f} (theoretical: 1.283)")
    print(f"  Min:  {min(samples):.3f}")
    print(f"  Max:  {max(samples):.3f}")

    # Check if mean is close to Euler-Mascheroni constant ≈ 0.577
    # With 10000 samples, standard error is ~0.0128, so within 0.05 should be fine
    assert abs(mean - 0.577) < 0.05, f"Mean {mean} too far from expected 0.577"

    # Check if std is close to π/√6 ≈ 1.283
    assert abs(std - 1.283) < 0.05, f"Std {std} too far from expected 1.283"

    print("\n✓ Gumbel sampling test PASSED!")
    return True


def test_gumbel_properties():
    """Test properties of Gumbel distribution."""
    print("\n" + "=" * 70)
    print("TEST: Gumbel Distribution Properties")
    print("=" * 70)

    # Test 1: Samples should be continuous and varied
    samples = [sample_gumbel() for _ in range(100)]
    unique_samples = len(set(samples))

    print(f"\nGenerated 100 samples")
    print(f"  Unique values: {unique_samples}")
    assert unique_samples > 95, "Samples should be diverse"

    # Test 2: Verify the transformation formula
    # G = -log(-log(U)) where U ~ Uniform(0,1)
    np.random.seed(42)
    u = np.random.uniform(0, 1, 5)
    expected_gumbel = -np.log(-np.log(u))

    np.random.seed(42)
    actual_gumbel = [sample_gumbel() for _ in range(5)]

    print(f"\nVerifying transformation formula:")
    for i, (e, a) in enumerate(zip(expected_gumbel, actual_gumbel)):
        print(f"  Sample {i+1}: expected={e:.4f}, actual={a:.4f}, diff={abs(e-a):.6f}")
        assert abs(e - a) < 1e-10, f"Transformation mismatch at index {i}"

    print("\n✓ Gumbel properties test PASSED!")
    return True


def test_gumbel_max_trick():
    """Test that argmax(utility + Gumbel) approximates exponential mechanism."""
    print("\n" + "=" * 70)
    print("TEST: Gumbel Max Trick Property")
    print("=" * 70)

    # Create some utilities
    utilities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    epsilon = 1.0
    alpha = 1.0

    # Run Gumbel max selection many times
    num_trials = 10000
    selection_counts = {i: 0 for i in range(len(utilities))}

    for _ in range(num_trials):
        # Sample Gumbel noise
        gumbel_noise = np.array([sample_gumbel() for _ in range(len(utilities))])

        # Compute scores: (ε/(2α)) · u + g
        scores = (epsilon / (2 * alpha)) * utilities + gumbel_noise

        # Select argmax
        selected = np.argmax(scores)
        selection_counts[selected] += 1

    # Compute expected probabilities using exponential mechanism
    # p(i) ∝ exp(ε · u_i / (2α))
    scaled_utilities = (epsilon / (2 * alpha)) * utilities
    exp_utilities = np.exp(scaled_utilities - np.max(scaled_utilities))
    expected_probs = exp_utilities / np.sum(exp_utilities)

    # Compute observed probabilities
    observed_probs = np.array([selection_counts[i] / num_trials for i in range(len(utilities))])

    print(f"\nRan {num_trials} trials with utilities {utilities}")
    print(f"\nExpected vs Observed probabilities:")
    for i in range(len(utilities)):
        exp_pct = expected_probs[i] * 100
        obs_pct = observed_probs[i] * 100
        diff = abs(exp_pct - obs_pct)
        print(f"  Option {i}: expected={exp_pct:.1f}%, observed={obs_pct:.1f}%, diff={diff:.1f}%")

    # Check that observed probabilities are close to expected (within 2%)
    max_diff = np.max(np.abs(expected_probs - observed_probs))
    print(f"\nMax difference: {max_diff*100:.1f}%")
    assert max_diff < 0.02, f"Observed probabilities differ too much from expected"

    print("\n✓ Gumbel max trick test PASSED!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("GUMBEL DELETION - SIMPLE TESTS")
    print("=" * 70)

    try:
        test_gumbel_sampling()
        test_gumbel_properties()
        test_gumbel_max_trick()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
