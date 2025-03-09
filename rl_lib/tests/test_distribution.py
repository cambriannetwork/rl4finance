"""
Tests for the distribution module.
"""

import unittest
import numpy as np

from rl_lib.distribution import Gaussian, Choose, Constant


class TestDistribution(unittest.TestCase):
    """Test cases for the distribution module."""
    
    def test_gaussian(self):
        """Test Gaussian distribution."""
        # Create a Gaussian distribution
        μ = 0.0
        σ = 1.0
        g = Gaussian(μ=μ, σ=σ)
        
        # Test sampling
        samples = g.sample_n(1000)
        self.assertEqual(len(samples), 1000)
        
        # Test mean and standard deviation (approximately)
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        self.assertAlmostEqual(sample_mean, μ, delta=0.1)
        self.assertAlmostEqual(sample_std, σ, delta=0.1)
        
        # Test expectation
        def f(x):
            return x ** 2
        
        # For a standard normal, E[X^2] = 1
        expectation = g.expectation(f)
        self.assertAlmostEqual(expectation, 1.0, delta=0.1)
    
    def test_choose(self):
        """Test Choose distribution."""
        # Create a Choose distribution
        options = [1, 2, 3, 4, 5]
        c = Choose(options)
        
        # Test sampling
        samples = c.sample_n(1000)
        self.assertEqual(len(samples), 1000)
        
        # Test that all samples are in the options
        for s in samples:
            self.assertIn(s, options)
        
        # Test expectation
        def f(x):
            return x
        
        # E[X] = (1 + 2 + 3 + 4 + 5) / 5 = 3
        expectation = c.expectation(f)
        self.assertEqual(expectation, 3.0)
    
    def test_constant(self):
        """Test Constant distribution."""
        # Create a Constant distribution
        value = 42
        c = Constant(value)
        
        # Test sampling
        samples = c.sample_n(10)
        self.assertEqual(len(samples), 10)
        
        # Test that all samples are equal to the value
        for s in samples:
            self.assertEqual(s, value)
        
        # Test expectation
        def f(x):
            return x * 2
        
        # E[2X] = 2 * 42 = 84
        expectation = c.expectation(f)
        self.assertEqual(expectation, 84)


if __name__ == '__main__':
    unittest.main()
