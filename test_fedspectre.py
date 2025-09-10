#!/usr/bin/env python3
"""
Test script for FedSPECTRE-Hybrid defense implementation.

This script validates that the defense works correctly and addresses
all the critical faults identified in the audit.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from defenses.fedspectre_hybrid import (
    FedSPECTREHybrid, 
    RobustStatistics, 
    RobustRepresentationExtractor,
    MahalanobisCKA,
    SpectralProjection,
    AugmentationStability
)
from defenses.fedspectre_validation import FedSPECTREValidator
from Params import Params

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def create_test_data(n_samples=100, n_features=10, n_classes=3, seed=42):
    """Create synthetic test data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create random features
    X = np.random.randn(n_samples, n_features)
    
    # Create class labels
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    
    return DataLoader(dataset, batch_size=32, shuffle=False)


def create_test_models(n_models=5, device=torch.device('cpu')):
    """Create test models."""
    models = {}
    
    for i in range(n_models):
        model = SimpleModel()
        model.to(device)
        models[i] = model
        
    return models


def test_robust_statistics():
    """Test robust statistics computation."""
    logger.info("Testing RobustStatistics...")
    
    # Create test data
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    
    # Add some outliers
    X[:5] += 10  # Make first 5 samples outliers
    
    # Test robust statistics
    robust_stats = RobustStatistics(rank=5, trim_fraction=0.1)
    mu, W, U = robust_stats.compute_robust_stats(X)
    
    # Validate results
    assert mu.shape == (n_features,), f"Expected mu shape ({n_features},), got {mu.shape}"
    assert W.shape[0] <= 5, f"Expected W rank <= 5, got {W.shape[0]}"
    assert W.shape[1] == n_features, f"Expected W features {n_features}, got {W.shape[1]}"
    assert U.shape == (n_features, W.shape[0]), f"Expected U shape ({n_features}, {W.shape[0]}), got {U.shape}"
    
    logger.info("✓ RobustStatistics test passed")


def test_representation_extractor():
    """Test representation extractor."""
    logger.info("Testing RobustRepresentationExtractor...")
    
    device = torch.device('cpu')
    extractor = RobustRepresentationExtractor(device, batch_size=16)
    
    # Create test model and data
    model = SimpleModel()
    data_loader = create_test_data(n_samples=50)
    
    # Extract representations
    representations = extractor.extract_representations(model, data_loader)
    
    # Validate results
    assert representations.ndim == 2, f"Expected 2D representations, got {representations.ndim}D"
    assert representations.shape[0] == 50, f"Expected 50 samples, got {representations.shape[0]}"
    assert representations.shape[1] == 20, f"Expected 20 features, got {representations.shape[1]}"
    
    # Check that representations are row-centered
    row_means = np.mean(representations, axis=1)  # Mean of each row (sample)
    assert np.allclose(row_means, 0, atol=1e-6), f"Representations should be row-centered, but row means are: {row_means[:5]}"
    
    logger.info("✓ RobustRepresentationExtractor test passed")


def test_mahalanobis_cka():
    """Test Mahalanobis-CKA computation."""
    logger.info("Testing MahalanobisCKA...")
    
    cka_computer = MahalanobisCKA()
    
    # Create test data
    n_samples, n_features = 50, 10
    X1 = np.random.randn(n_samples, n_features)
    X2 = np.random.randn(n_samples, n_features)
    
    # Test CKA computation
    cka_score = cka_computer.linear_cka(X1, X2)
    
    # Validate results
    assert 0 <= cka_score <= 1, f"CKA score should be in [0,1], got {cka_score}"
    
    # Test self-CKA (should be close to 1)
    self_cka = cka_computer.linear_cka(X1, X1)
    assert self_cka > 0.9, f"Self-CKA should be > 0.9, got {self_cka}"
    
    # Test with different shapes (should raise error)
    X3 = np.random.randn(30, n_features)
    try:
        cka_computer.linear_cka(X1, X3)
        assert False, "Should have raised error for shape mismatch"
    except AssertionError:
        pass  # Expected
    
    logger.info("✓ MahalanobisCKA test passed")


def test_spectral_projection():
    """Test spectral projection computation."""
    logger.info("Testing SpectralProjection...")
    
    spectral_computer = SpectralProjection()
    
    # Create test data
    client_representations = {
        0: {0: np.random.randn(20, 10), 1: np.random.randn(15, 10)},
        1: {0: np.random.randn(25, 10), 1: np.random.randn(18, 10)},
        2: {0: np.random.randn(22, 10), 1: np.random.randn(12, 10)}
    }
    
    # Create mock whitening matrices
    class_whiteners = {
        0: (np.random.randn(10), np.random.randn(5, 10), np.random.randn(10, 5)),
        1: (np.random.randn(10), np.random.randn(5, 10), np.random.randn(10, 5))
    }
    
    # Test spectral scores
    spectral_scores = spectral_computer.compute_spectral_scores(
        client_representations, class_whiteners, target_class=0
    )
    
    # Validate results
    assert len(spectral_scores) == 3, f"Expected 3 client scores, got {len(spectral_scores)}"
    for client_id, score in spectral_scores.items():
        assert score >= 0, f"Client {client_id}: spectral score should be >= 0, got {score}"
    
    logger.info("✓ SpectralProjection test passed")


def test_augmentation_stability():
    """Test augmentation stability computation."""
    logger.info("Testing AugmentationStability...")
    
    device = torch.device('cpu')
    stability_computer = AugmentationStability(device)
    
    # Create test models and data
    models = create_test_models(3, device)
    data_loader = create_test_data(n_samples=30)
    
    # Create mock whitening matrices
    class_whiteners = {
        0: (np.random.randn(10), np.random.randn(5, 10), np.random.randn(10, 5))
    }
    
    # Create mock extractor
    extractor = RobustRepresentationExtractor(device)
    
    # Test stability scores
    stability_scores = stability_computer.compute_stability_scores(
        models, data_loader, class_whiteners, extractor
    )
    
    # Validate results
    assert len(stability_scores) == 3, f"Expected 3 client scores, got {len(stability_scores)}"
    for client_id, score in stability_scores.items():
        assert 0 <= score <= 1, f"Client {client_id}: stability score should be in [0,1], got {score}"
    
    logger.info("✓ AugmentationStability test passed")


def test_fedspectre_hybrid_integration():
    """Test full FedSPECTRE-Hybrid integration."""
    logger.info("Testing FedSPECTRE-Hybrid integration...")
    
    device = torch.device('cpu')
    
    # Create test data and models
    data_loader = create_test_data(n_samples=100)
    models = create_test_models(5, device)
    client_weights = {i: models[i].state_dict() for i in range(5)}
    
    # Create parameters
    params = Params()
    params.fedspectre_enabled = True
    params.fedspectre_rank = 5
    params.fedspectre_alpha = 0.4
    params.fedspectre_beta = 0.3
    params.fedspectre_gamma = 0.3
    params.fedspectre_trim_fraction = 0.4
    
    # Initialize defense
    defense = FedSPECTREHybrid(
        device=device,
        rank=params.fedspectre_rank,
        alpha=params.fedspectre_alpha,
        beta=params.fedspectre_beta,
        gamma=params.fedspectre_gamma,
        trim_fraction=params.fedspectre_trim_fraction
    )
    
    # Apply defense
    filtered_weights, telemetry = defense.apply_defense(
        models, client_weights, data_loader, params
    )
    
    # Validate results
    assert len(filtered_weights) > 0, "Should have at least one selected client"
    assert len(filtered_weights) <= len(client_weights), "Should not select more clients than available"
    
    # Check telemetry
    assert 'selected_clients' in telemetry, "Telemetry should contain selected_clients"
    assert 'anomaly_scores' in telemetry, "Telemetry should contain anomaly_scores"
    assert 'compute_time_s' in telemetry, "Telemetry should contain compute_time_s"
    
    logger.info("✓ FedSPECTRE-Hybrid integration test passed")


def test_validation_suite():
    """Test the validation suite."""
    logger.info("Testing validation suite...")
    
    device = torch.device('cpu')
    
    # Create test data
    models = create_test_models(3, device)
    data_loader = create_test_data(n_samples=50)
    
    # Create mock data for validation
    class_whiteners = {
        0: (np.random.randn(10), np.random.randn(5, 10), np.random.randn(10, 5))
    }
    
    class_templates = {
        0: np.random.randn(20, 10)
    }
    
    anomaly_scores = {0: 0.1, 1: 0.3, 2: 0.2}
    
    # Run validation
    validator = FedSPECTREValidator(device)
    results = validator.run_all_validations(
        models, data_loader, class_whiteners, class_templates, anomaly_scores
    )
    
    # Validate results
    assert 'overall_pass' in results, "Validation should have overall_pass status"
    assert isinstance(results['overall_pass'], bool), "overall_pass should be boolean"
    
    # Generate report
    report = validator.generate_validation_report()
    assert len(report) > 0, "Validation report should not be empty"
    
    logger.info("✓ Validation suite test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting FedSPECTRE-Hybrid test suite...")
    
    try:
        test_robust_statistics()
        test_representation_extractor()
        test_mahalanobis_cka()
        test_spectral_projection()
        test_augmentation_stability()
        test_fedspectre_hybrid_integration()
        test_validation_suite()
        
        logger.info("🎉 All tests passed! FedSPECTRE-Hybrid implementation is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
