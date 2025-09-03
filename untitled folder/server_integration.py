"""
Enhanced Server.py with FedAvgCKA defense integration.

This update adds FedAvgCKA capabilities to the existing ServerAvg class.
"""

import copy
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch import optim, nn, kl_div"""
Server-side integration for FedAvgCKA defense.

This module provides the necessary modifications to integrate FedAvgCKA 
into the existing ServerAvg class without breaking existing functionality.
"""

import copy
import logging
from collections import OrderedDict
from typing import Dict, List, Any, Tuple

import torch
from torch.utils.data import DataLoader

from defenses.fedavgcka import (
    create_root_dataset, 
    apply_fedavgcka_filter
)

logger = logging.getLogger(__name__)


class FedAvgCKAServerMixin:
    """
    Mixin class to add FedAvgCKA functionality to ServerAvg.
    
    This mixin provides FedAvgCKA defense capabilities that can be added
    to the existing ServerAvg class without modifying the original code.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dataset_loader = None
        self.fedavgcka_telemetry = []
        
    def initialize_fedavgcka(self, task, params):
        """
        Initialize FedAvgCKA components (root dataset, etc.).
        
        Args:
            task: FederatedTask instance 
            params: Params object with FedAvgCKA configuration
            
        Note:
            Should be called once during server initialization.
        """
        if not params.fedavgcka_enabled:
            return
            
        try:
            logger.info("Initializing FedAvgCKA defense...")
            
            # Create root dataset for activation extraction
            self.root_dataset_loader = create_root_dataset(
                task=task,
                size=params.fedavgcka_root_dataset_size,
                strategy=params.fedavgcka_root_dataset_strategy,
                device=self.device
            )
            
            logger.info(f"FedAvgCKA initialized with root dataset of size {params.fedavgcka_root_dataset_size}")
            logger.info(f"Root dataset strategy: {params.fedavgcka_root_dataset_strategy}")
            logger.info(f"Layer comparison mode: {params.fedavgcka_layer_comparison}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FedAvgCKA: {e}")
            params.fedavgcka_enabled = False  # Disable on failure
            
    def fedavgcka_aggregate_global_model(self, clients, chosen_ids, task, params):
        """
        Aggregate global model using FedAvgCKA pre-filtering.
        
        Args:
            clients: List of Client objects
            chosen_ids: List of client IDs selected for this round
            task: FederatedTask instance
            params: Params object with FedAvgCKA configuration
            
        Note:
            This is a drop-in replacement for aggregate_global_model()
            when FedAvgCKA defense is enabled.
        """
        if not params.fedavgcka_enabled or self.root_dataset_loader is None:
            # Fallback to standard aggregation
            logger.warning("FedAvgCKA not properly initialized, falling back to standard aggregation")
            return self._standard_aggregate(clients, chosen_ids)
        
        try:
            # Collect client models and weights
            client_models = {}
            client_weights = {}
            
            for client_id in chosen_ids:
                client = clients[client_id]
                client_models[client_id] = copy.deepcopy(client.local_model)
                client_weights[client_id] = client.local_model.state_dict()
            
            logger.info(f"Applying FedAvgCKA filtering to {len(chosen_ids)} clients...")
            
            # Apply FedAvgCKA filtering
            filtered_weights, telemetry = apply_fedavgcka_filter(
                client_models=client_models,
                client_weights=client_weights,
                params=params,
                root_loader=self.root_dataset_loader,
                device=self.device
            )
            
            # Store telemetry for analysis
            telemetry['round'] = getattr(self, 'current_round', -1)
            telemetry['original_clients'] = chosen_ids
            self.fedavgcka_telemetry.append(telemetry)
            
            if not filtered_weights:
                logger.error("FedAvgCKA filtering resulted in no clients! Falling back to original.")
                filtered_weights = client_weights
            
            # Perform FedAvg aggregation on filtered weights
            self._aggregate_filtered_weights(filtered_weights, clients)
            
            logger.info(f"FedAvgCKA aggregation complete. Used {len(filtered_weights)}/{len(chosen_ids)} clients.")
            
        except Exception as e:
            logger.error(f"FedAvgCKA aggregation failed: {e}. Falling back to standard aggregation.")
            self._standard_aggregate(clients, chosen_ids)
    
    def _aggregate_filtered_weights(self, filtered_weights: Dict[int, OrderedDict], clients):
        """
        Perform FedAvg aggregation using only the filtered client weights.
        
        Args:
            filtered_weights: Dict of client_id -> state_dict for selected clients
            clients: List of all Client objects (for sample counts)
        """
        if not filtered_weights:
            logger.error("No weights to aggregate!")
            return
        
        # Calculate weighted average based on data size
        averaged_weights = OrderedDict()
        total_samples = 0
        
        # Initialize structure from first client
        first_client_weights = next(iter(filtered_weights.values()))
        for layer_name in first_client_weights.keys():
            averaged_weights[layer_name] = torch.zeros_like(first_client_weights[layer_name])
        
        # Calculate total samples from selected clients
        for client_id in filtered_weights.keys():
            total_samples += clients[client_id].n_sample
        
        # Weighted aggregation
        for client_id, weights in filtered_weights.items():
            client_weight = clients[client_id].n_sample / total_samples
            
            for layer_name, layer_weights in weights.items():
                averaged_weights[layer_name] += client_weight * layer_weights
        
        # Update global model
        self.global_model.load_state_dict(averaged_weights)
        
    def _standard_aggregate(self, clients, chosen_ids):
        """
        Fallback to standard FedAvg aggregation.
        
        Args:
            clients: List of Client objects
            chosen_ids: List of client IDs
        """
        # This should call the original aggregate_global_model method
        # Implementation depends on the existing ServerAvg structure
        averaged_weights = OrderedDict()
        for layer, weight in self.global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        total_prop = sum(clients[client_id].n_sample for client_id in chosen_ids)

        for client_id in chosen_ids:
            client = clients[client_id]
            prop = client.n_sample / total_prop
            
            for layer, weight in client.local_model.state_dict().items():
                averaged_weights[layer] += prop * weight

        self.global_model.load_state_dict(averaged_weights)
        
    def get_fedavgcka_telemetry(self) -> List[Dict[str, Any]]:
        """
        Get collected FedAvgCKA telemetry data.
        
        Returns:
            List of telemetry dictionaries from each round
        """
        return self.fedavgcka_telemetry.copy()
        
    def reset_fedavgcka_telemetry(self):
        """Reset collected telemetry data."""
        self.fedavgcka_telemetry = []


# Example of how to integrate with existing ServerAvg class
"""
To integrate FedAvgCKA into the existing ServerAvg class, you can either:

1. Add the mixin to ServerAvg:
   class ServerAvg(ServerBase, FedAvgCKAServerMixin):
       ...

2. Or create a new server class:
   class ServerAvgWithCKA(ServerAvg, FedAvgCKAServerMixin):
       ...

Then modify the training loop to:
- Call initialize_fedavgcka() during setup
- Use fedavgcka_aggregate_global_model() instead of aggregate_global_model()
  when FedAvgCKA is enabled
"""


def create_fedavgcka_server(base_server_class):
    """
    Factory function to create a server class with FedAvgCKA capabilities.
    
    Args:
        base_server_class: The base server class (e.g., ServerAvg)
        
    Returns:
        New server class with FedAvgCKA mixin
    """
    class FedAvgCKAServer(base_server_class, FedAvgCKAServerMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def aggregate_global_model(self, clients, chosen_ids, task=None, params=None):
            """Override to use FedAvgCKA when enabled."""
            if hasattr(params, 'fedavgcka_enabled') and params.fedavgcka_enabled:
                return self.fedavgcka_aggregate_global_model(clients, chosen_ids, task, params)
            else:
                return super().aggregate_global_model(clients, chosen_ids, None)
    
    return FedAvgCKAServer