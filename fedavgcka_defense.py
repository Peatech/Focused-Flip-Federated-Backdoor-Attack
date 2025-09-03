"""
FedAvgCKA Defense Implementation

Complete implementation of the FedAvgCKA defense for federated learning
as described in "Exploiting Layerwise Feature Representation Similarity 
For Backdoor Defence in Federated Learning" by Walter et al. (ESORICS 2024)

This module integrates directly with the existing federated learning codebase.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import logging
import copy
import time

logger = logging.getLogger(__name__)


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute linear CKA similarity between two activation matrices.
    
    Based on FedAvgCKA Algorithm 1 and Equation 4.
    
    Args:
        X: Activation matrix [k, d1] 
        Y: Activation matrix [k, d2] 
        eps: Small constant for numerical stability
        
    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], f"Batch size mismatch: {X.shape[0]} vs {Y.shape[0]}"
    
    n = X.shape[0]
    if n <= 1:
        logger.warning(f"CKA computation with n={n} samples may be unreliable")
        return 0.0
    
    # Center the matrices (H = I - (1/n)*11^T)
    H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
    
    # Apply centering: X_centered = XH, Y_centered = YH  
    X_centered = torch.mm(X, H)
    Y_centered = torch.mm(Y, H)
    
    # Compute HSIC values using linear kernel
    hsic_xy = torch.trace(torch.mm(X_centered, Y_centered.t())) / ((n - 1) ** 2)
    hsic_xx = torch.trace(torch.mm(X_centered, X_centered.t())) / ((n - 1) ** 2)  
    hsic_yy = torch.trace(torch.mm(Y_centered, Y_centered.t())) / ((n - 1) ** 2)
    
    # Compute normalized CKA score
    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < eps:
        logger.warning(f"Small CKA denominator: {denominator}. Returning 0.0")
        return 0.0
        
    cka_score = hsic_xy / denominator
    cka_score = torch.clamp(cka_score, 0.0, 1.0)
    
    return float(cka_score)


def get_layer_activations(
    model: nn.Module, 
    data_loader: DataLoader,
    layer_name: str,
    device: torch.device
) -> torch.Tensor:
    """
    Extract activations from a specific layer of the model.
    
    Args:
        model: Neural network model (will be set to eval mode)
        data_loader: DataLoader containing input data
        layer_name: Name of layer to extract activations from
        device: Device to run computations on
        
    Returns:
        Activation matrix of shape [n_samples, activation_dim]
    """
    model.eval()
    model.to(device)
    
    activations = []
    
    def hook_fn(module, input, output):
        # Flatten spatial dimensions if needed
        if output.dim() > 2:
            output = output.view(output.size(0), -1)
        activations.append(output.detach().cpu())
    
    # Find and register hook
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        available_layers = [name for name, _ in model.named_modules()]
        raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0].to(device)
                else:
                    inputs = batch_data.to(device)
                
                _ = model(inputs)
        
        if not activations:
            raise RuntimeError("No activations captured. Check layer name and data loader.")
            
        activation_matrix = torch.cat(activations, dim=0)
        
        # Row-center the activation matrix
        activation_matrix = activation_matrix - activation_matrix.mean(dim=0, keepdim=True)
        
        return activation_matrix
        
    finally:
        handle.remove()


def get_penultimate_layer_name(model: nn.Module) -> str:
    """Automatically determine the penultimate layer name."""
    # Check for ResNet architecture
    if hasattr(model, 'fc') and hasattr(model, 'avgpool'):
        return 'avgpool'
    
    # Check for SimpleNet architecture  
    if hasattr(model, 'fc2') and hasattr(model, 'fc1'):
        return 'fc1'
        
    # Check for VGG-like architectures
    if hasattr(model, 'classifier') and hasattr(model, 'features'):
        return 'features'
    
    # Generic fallback
    layers = list(model.named_modules())
    if len(layers) < 2:
        raise ValueError("Model too simple to determine penultimate layer")
        
    penultimate_name = layers[-2][0]
    logger.info(f"Auto-detected penultimate layer: '{penultimate_name}'")
    return penultimate_name


def create_root_dataset(
    task, 
    size: int = 16,
    strategy: str = "class_balanced",
    device: torch.device = None
) -> DataLoader:
    """Create root dataset R from task's test data."""
    if device is None:
        device = task.params.device
        
    if size <= 0:
        raise ValueError(f"Root dataset size must be positive, got {size}")
        
    if strategy not in ["random", "class_balanced"]:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    test_dataset = task.test_dataset
    total_samples = len(test_dataset)
    
    if size > total_samples:
        logger.warning(f"Requested size {size} > available samples {total_samples}. Using all samples.")
        size = total_samples
    
    if strategy == "random":
        indices = np.random.choice(total_samples, size=size, replace=False)
        
    elif strategy == "class_balanced":
        try:
            # Group samples by class
            class_to_indices = defaultdict(list)
            for idx in range(total_samples):
                _, label = test_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_to_indices[label].append(idx)
            
            n_classes = len(class_to_indices)
            samples_per_class = size // n_classes
            remaining_samples = size % n_classes
            
            indices = []
            for class_idx, (label, class_indices) in enumerate(class_to_indices.items()):
                class_size = samples_per_class + (1 if class_idx < remaining_samples else 0)
                class_size = min(class_size, len(class_indices))
                
                selected = np.random.choice(class_indices, size=class_size, replace=False)
                indices.extend(selected)
            
            logger.info(f"Created class-balanced root dataset: {len(indices)} samples across {n_classes} classes")
            
        except Exception as e:
            logger.warning(f"Class-balanced sampling failed: {e}. Falling back to random sampling.")
            indices = np.random.choice(total_samples, size=size, replace=False)
    
    root_subset = Subset(test_dataset, indices)
    root_loader = DataLoader(
        root_subset, 
        batch_size=size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Created root dataset with {len(indices)} samples using '{strategy}' strategy")
    return root_loader


def get_layer_names_for_comparison(model: nn.Module, layer_comparison: str) -> List[str]:
    """Get layer names based on comparison configuration."""
    if layer_comparison == "penultimate":
        return [get_penultimate_layer_name(model)]
    
    elif layer_comparison == "layer2":
        if hasattr(model, 'layer2'):
            return ['layer2']
        elif hasattr(model, 'conv2'):
            return ['conv2']
        else:
            raise ValueError("Model does not have layer2 or conv2 for layer2 comparison")
            
    elif layer_comparison == "layer3":
        if hasattr(model, 'layer3'):
            return ['layer3']
        elif hasattr(model, 'fc1'):
            return ['fc1']
        else:
            raise ValueError("Model does not have layer3 or fc1 for layer3 comparison")
            
    elif layer_comparison == "multi_layer":
        layers = []
        
        try:
            layers.append(get_penultimate_layer_name(model))
        except ValueError:
            logger.warning("Could not find penultimate layer for multi-layer comparison")
        
        if hasattr(model, 'layer3'):
            layers.append('layer3')
        elif hasattr(model, 'fc1'):
            layers.append('fc1')
            
        if hasattr(model, 'layer2'):
            layers.append('layer2')
        elif hasattr(model, 'conv2'):
            layers.append('conv2')
        
        if not layers:
            raise ValueError("Could not find any suitable layers for multi-layer comparison")
            
        return layers
    
    else:
        raise ValueError(f"Unknown layer_comparison option: {layer_comparison}")


def rank_clients_by_cka(
    activations: Dict[int, torch.Tensor],
    trim_fraction: float = 0.5
) -> Tuple[List[int], List[int], Dict[int, float]]:
    """Rank clients by CKA similarity and determine exclusions."""
    if not activations:
        return [], [], {}
        
    client_ids = list(activations.keys())
    n_clients = len(client_ids)
    
    if n_clients == 1:
        return client_ids, [], {client_ids[0]: 1.0}
    
    # Compute pairwise CKA scores
    cka_matrix = torch.zeros(n_clients, n_clients)
    
    logger.info(f"Computing pairwise CKA scores for {n_clients} clients...")
    
    for i in range(n_clients):
        for j in range(i, n_clients):
            if i == j:
                cka_score = 1.0
            else:
                client_i, client_j = client_ids[i], client_ids[j]
                cka_score = linear_cka(activations[client_i], activations[client_j])
            
            cka_matrix[i, j] = cka_score
            cka_matrix[j, i] = cka_score
    
    # Calculate average CKA score for each client
    avg_cka_scores = {}
    for i, client_id in enumerate(client_ids):
        if n_clients > 1:
            mask = torch.ones(n_clients, dtype=torch.bool)
            mask[i] = False
            avg_score = cka_matrix[i, mask].mean().item()
        else:
            avg_score = 1.0
        avg_cka_scores[client_id] = avg_score
    
    # Sort clients by average CKA score (ascending order)
    sorted_clients = sorted(client_ids, key=lambda cid: avg_cka_scores[cid])
    
    # Determine exclusions
    n_exclude = int(trim_fraction * n_clients)
    n_exclude = min(n_exclude, n_clients - 1)
    
    excluded_clients = sorted_clients[:n_exclude]
    selected_clients = sorted_clients[n_exclude:]
    
    logger.info(f"CKA ranking: selected {len(selected_clients)}, excluded {len(excluded_clients)}")
    return selected_clients, excluded_clients, avg_cka_scores


def apply_fedavgcka_filter(
    client_models: Dict[int, nn.Module],
    client_weights: Dict[int, Any],
    params,
    root_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """Apply FedAvgCKA filtering before aggregation."""
    if not client_models:
        return {}, {"error": "No client models provided"}
    
    start_time = time.time()
    
    try:
        sample_model = next(iter(client_models.values()))
        layer_names = get_layer_names_for_comparison(sample_model, params.fedavgcka_layer_comparison)
        
        client_ids = list(client_models.keys())
        
        if params.fedavgcka_layer_comparison == "multi_layer":
            # Multi-layer comparison
            combined_scores = compute_multi_layer_cka_scores(
                client_models, 
                root_loader, 
                layer_names,
                params.fedavgcka_multi_layer_weights,
                device
            )
            
            sorted_clients = sorted(client_ids, key=lambda cid: combined_scores.get(cid, 0.0))
            
            n_exclude = int(params.fedavgcka_trim_fraction * len(client_ids))
            n_exclude = min(n_exclude, len(client_ids) - 1)
            
            excluded_clients = sorted_clients[:n_exclude]
            selected_clients = sorted_clients[n_exclude:]
            cka_scores = combined_scores
            
        else:
            # Single layer comparison
            layer_name = layer_names[0]
            logger.info(f"Extracting activations from layer: {layer_name}")
            
            activations = {}
            failed_clients = []
            
            for client_id, model in client_models.items():
                try:
                    client_activations = get_layer_activations(model, root_loader, layer_name, device)
                    activations[client_id] = client_activations
                except Exception as e:
                    logger.error(f"Failed to extract activations for client {client_id}: {e}")
                    failed_clients.append(client_id)
            
            if len(activations) < 2:
                logger.warning("Too few clients with valid activations. Skipping FedAvgCKA filtering.")
                return client_weights, {
                    "error": "Insufficient valid activations",
                    "failed_clients": failed_clients,
                    "selected_clients": list(client_weights.keys()),
                    "excluded_clients": []
                }
            
            selected_clients, excluded_clients, cka_scores = rank_clients_by_cka(
                activations, 
                params.fedavgcka_trim_fraction
            )
        
        # Filter client weights
        filtered_weights = {
            client_id: client_weights[client_id] 
            for client_id in selected_clients 
            if client_id in client_weights
        }
        
        compute_time = time.time() - start_time
        
        telemetry = {
            "selected_clients": selected_clients,
            "excluded_clients": excluded_clients,
            "cka_scores": cka_scores,
            "layer_names": layer_names,
            "n_selected": len(selected_clients),
            "n_excluded": len(excluded_clients), 
            "trim_fraction": params.fedavgcka_trim_fraction,
            "compute_time_s": compute_time
        }
        
        if params.fedavgcka_log_scores:
            logger.info(f"FedAvgCKA filtering complete:")
            logger.info(f"  Selected: {len(selected_clients)} clients {selected_clients}")
            logger.info(f"  Excluded: {len(excluded_clients)} clients {excluded_clients}")
            logger.info(f"  Compute time: {compute_time:.2f} s")
        
        return filtered_weights, telemetry
        
    except Exception as e:
        logger.error(f"FedAvgCKA filtering failed: {e}")
        return client_weights, {
            "error": str(e),
            "selected_clients": list(client_weights.keys()),
            "excluded_clients": []
        }


def compute_multi_layer_cka_scores(
    client_models: Dict[int, nn.Module],
    root_loader: DataLoader,
    layer_names: List[str],
    layer_weights: Dict[str, float],
    device: torch.device
) -> Dict[int, float]:
    """Compute weighted combination of CKA scores across multiple layers."""
    client_ids = list(client_models.keys())
    combined_scores = {client_id: 0.0 for client_id in client_ids}
    
    logger.info(f"Computing multi-layer CKA scores for layers: {layer_names}")
    
    for layer_name in layer_names:
        logger.info(f"Processing layer: {layer_name}")
        
        layer_activations = {}
        for client_id, model in client_models.items():
            try:
                activations = get_layer_activations(model, root_loader, layer_name, device)
                layer_activations[client_id] = activations
            except Exception as e:
                logger.error(f"Failed to extract activations from {layer_name} for client {client_id}: {e}")
                continue
        
        if len(layer_activations) < 2:
            logger.warning(f"Too few clients ({len(layer_activations)}) have valid activations for {layer_name}, skipping")
            continue
        
        _, _, layer_cka_scores = rank_clients_by_cka(layer_activations, trim_fraction=0.0)
        
        weight = layer_weights.get(layer_name, 1.0 / len(layer_names))
        
        for client_id in layer_cka_scores:
            if client_id in combined_scores:
                combined_scores[client_id] += weight * layer_cka_scores[client_id]
    
    logger.info(f"Multi-layer CKA computation complete. Combined scores: {combined_scores}")
    return combined_scores
