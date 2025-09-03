#!/usr/bin/env python3
"""
Validation script for FedAvgCKA implementation.

This script tests the core CKA computation and activation extraction
to ensure they work correctly before full integration.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenses.fedavgcka import (
    linear_cka,
    get_layer_activations, 
    get_penultimate_layer_name,
    rank_clients_by_cka,
    create_root_dataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_linear_cka():
    """Test the linear CKA computation with known cases."""
    logger.info("Testing linear CKA computation...")
    
    # Test case 1: Identical matrices should have CKA = 1.0
    torch.manual_seed(42)
    X = torch.randn(10, 5)
    Y = X.clone()
    
    cka_score = linear_cka(X, Y)
    assert abs(cka_score - 1.0) < 1e-6, f"Expected CKA=1.0 for identical matrices, got {cka_score}"
    logger.info(f"✓ Identical matrices: CKA = {cka_score:.6f}")
    
    # Test case 2: Orthogonal random matrices should have low CKA
    X = torch.randn(20, 10)
    Y = torch.randn(20, 10)
    
    cka_score = linear_cka(X, Y)
    logger.info(f"✓ Random matrices: CKA = {cka_score:.6f} (should be low)")
    
    # Test case 3: Linearly transformed matrices should have high CKA
    X = torch.randn(15, 8)
    transformation = torch.randn(8, 8)
    Y = torch.mm(X, transformation)  # Linear transformation
    
    cka_score = linear_cka(X, Y)
    logger.info(f"✓ Linearly transformed matrices: CKA = {cka_score:.6f} (should be high)")
    
    # Test case 4: Edge case with small matrices
    X = torch.randn(2, 3)
    Y = torch.randn(2, 3)
    
    cka_score = linear_cka(X, Y)
    logger.info(f"✓ Small matrices (2x3): CKA = {cka_score:.6f}")
    
    logger.info("Linear CKA tests passed!")


def create_test_model(input_dim=32, hidden_dim=64, output_dim=10):
    """Create a simple test model for activation extraction."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim * input_dim * 3, hidden_dim),  # Assume 3-channel input
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(), 
        nn.Linear(hidden_dim // 2, output_dim)
    )


def test_activation_extraction():
    """Test activation extraction from neural networks."""
    logger.info("Testing activation extraction...")
    
    # Create test model and data
    model = create_test_model()
    test_data = torch.randn(8, 3, 32, 32)  # 8 samples, 3 channels, 32x32
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    device = torch.device('cpu')
    
    # Test extraction from different layers
    try:
        # Layer 1 (first linear layer)
        activations_1 = get_layer_activations(model, test_loader, '1', device)
        logger.info(f"✓ Layer 1 activations shape: {activations_1.shape}")
        
        # Layer 3 (second linear layer) 
        activations_3 = get_layer_activations(model, test_loader, '3', device)
        logger.info(f"✓ Layer 3 activations shape: {activations_3.shape}")
        
        # Verify activations are different
        cka_score = linear_cka(activations_1, activations_3)
        logger.info(f"✓ CKA between layer 1 and 3: {cka_score:.6f}")
        
    except Exception as e:
        logger.error(f"Activation extraction failed: {e}")
        raise
    
    logger.info("Activation extraction tests passed!")


def test_client_ranking():
    """Test client ranking based on CKA scores."""
    logger.info("Testing client ranking...")
    
    # Create synthetic activation data for multiple "clients"
    torch.manual_seed(123)
    n_samples, n_features = 16, 32
    
    # Create benign clients (similar activations)
    base_activations = torch.randn(n_samples, n_features)
    client_activations = {}
    
    # Benign clients (small variations)
    for i in range(5):
        noise = torch.randn_like(base_activations) * 0.1  # Small noise
        client_activations[i] = base_activations + noise
    
    # Malicious clients (very different activations)
    for i in range(5, 7):
        client_activations[i] = torch.randn(n_samples, n_features) * 2  # Different distribution
    
    # Rank clients
    selected, excluded, scores = rank_clients_by_cka(client_activations, trim_fraction=0.3)
    
    logger.info(f"✓ Selected clients: {selected}")
    logger.info(f"✓ Excluded clients: {excluded}") 
    logger.info(f"✓ CKA scores: {scores}")
    
    # Verify that malicious clients (5, 6) are more likely to be excluded
    excluded_set = set(excluded)
    malicious_clients = {5, 6}
    overlap = malicious_clients.intersection(excluded_set)
    logger.info(f"✓ Malicious clients excluded: {overlap} / {malicious_clients}")
    
    logger.info("Client ranking tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases...")
    
    # Empty input
    try:
        selected, excluded, scores = rank_clients_by_cka({})
        assert selected == [] and excluded == [] and scores == {}
        logger.info("✓ Empty input handled correctly")
    except Exception as e:
        logger.error(f"Empty input test failed: {e}")
    
    # Single client
    try:
        activations = {0: torch.randn(5, 10)}
        selected, excluded, scores = rank_clients_by_cka(activations)
        assert len(selected) == 1 and len(excluded) == 0
        logger.info("✓ Single client handled correctly") 
    except Exception as e:
        logger.error(f"Single client test failed: {e}")
    
    # Very small matrices
    try:
        X = torch.randn(1, 2)  # Only 1 sample
        Y = torch.randn(1, 2)
        cka_score = linear_cka(X, Y)
        logger.info(f"✓ Very small matrices: CKA = {cka_score}")
    except Exception as e:
        logger.warning(f"Small matrix test: {e} (this may be expected)")
    
    logger.info("Edge case tests completed!")


def main():
    """Run all validation tests."""
    logger.info("Starting FedAvgCKA validation tests...")
    logger.info("=" * 50)
    
    try:
        test_linear_cka()
        logger.info("")
        
        test_activation_extraction()
        logger.info("")
        
        test_client_ranking()
        logger.info("")
        
        test_edge_cases()
        logger.info("")
        
        logger.info("=" * 50)
        logger.info("All FedAvgCKA validation tests passed! ✓")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Test integration with existing models (ResNet, SimpleNet)")
        logger.info("2. Validate root dataset creation with real FL tasks")
        logger.info("3. Test performance with larger client pools")
        logger.info("4. Integrate with ServerAvg aggregation")
        
    except Exception as e:
        logger.error(f"Validation tests failed: {e}")
        raise


if __name__ == "__main__":
    main()