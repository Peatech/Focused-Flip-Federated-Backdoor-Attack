    def get_fedavgcka_telemetry(self):
        """Get FedAvgCKA telemetry data."""
        return self.fedavgcka_telemetry.copy()
        
    def reset_fedavgcka_telemetry(self):
        """Reset FedAvgCKA telemetry."""
        self.fedavgcka_telemetry = []

    def collect_conv_ranks(self, task, clients: client_group, chosen_ids, pts):
        client_ranks = list()
        for id in chosen_ids:
            client = clients[id]
            client_ranks.append(client.get_conv_rank(task, n_test_batch=10, location='last'))

        averaged_client_rank = torch.zeros_like(client_ranks[0])
        for client_rank in client_ranks:
            averaged_client_rank = averaged_client_rank + client_rank
        averaged_client_rank = averaged_client_rank / len(client_ranks)
        _, prune_orders = torch.sort(averaged_client_rank, descending=True)
        prune_orders = prune_orders.cpu().numpy().tolist()
        print("prune_orders:", prune_orders)
        return prune_orders

    def conv_pruning(self, task, orders):
        self.global_model.eval()
        model_weights = self.global_model.state_dict()
        final_conv = get_conv_weight_names(self.global_model)[-1]

        final_gamma, final_bias = None, None
        if isinstance(self.global_model, ResNet):
            final_gamma = final_conv.replace("conv", "bn")
            final_bias = final_conv.replace("conv", "bn").replace('weight', 'bias')

        last_accuracy = get_accuracy(self.global_model, task, self.train_loader)
        for i, conv_id in enumerate(orders):
            original_weights = self.global_model.state_dict()

            model_weights[final_conv][conv_id] = torch.zeros_like(model_weights[final_conv][conv_id])
            if final_bias is not None and final_gamma is not None:
                model_weights[final_gamma][conv_id] = 0.0
                model_weights[final_bias][conv_id] = 0.0

            self.global_model.load_state_dict(model_weights)
            current_accuracy = get_accuracy(self.global_model, task, self.train_loader)

            if last_accuracy - current_accuracy >= 0.01:
                self.global_model.load_state_dict(original_weights)
                print("prune:{}".format(i))
                return

    def adjust_extreme_parameters(self, threshold):
        self.global_model.eval()
        model_weights = self.global_model.state_dict()
        final_conv = get_conv_weight_names(self.global_model)[-1]
        min_w = float(torch.mean(model_weights[final_conv]) - threshold * torch.std(model_weights[final_conv]))
        max_w = float(torch.mean(model_weights[final_conv]) + threshold * torch.std(model_weights[final_conv]))
        model_weights[final_conv][model_weights[final_conv] > max_w]= 0.0
        model_weights[final_conv][model_weights[final_conv] < min_w]= 0.0
        p_zero = torch.sum((model_weights[final_conv] == 0.0).int()).item() / model_weights[final_conv].numel()
        print("Adjust Extreme Value: {}".format(p_zero))
        self.global_model.load_state_dict(model_weights)

    def sign_voting_aggregate_global_model(self, clients: client_group, chosen_ids, pts):
        assert not clients is None and len(clients) > 0
        original_params = self.global_model.state_dict()

        total_sample = 0
        for id in chosen_ids:
            client = clients[id]
            total_sample = total_sample + client.n_sample

        # collect client updates
        updates = list()
        for id in chosen_ids:
            client = clients[id]
            local_params = client.local_model.state_dict()
            update = OrderedDict()
            for layer, weight in local_params.items():
                update[layer] = local_params[layer] - original_params[layer]
            updates.append(update)

        # compute_total_update
        robust_lrs = self.compute_robustLR(updates)
        # count signs：
        flip_analysis = dict()
        for layer in robust_lrs.keys():
            n_flip = torch.sum(torch.gt(robust_lrs[layer], 0.0).int())
            n_unflip = torch.sum(torch.lt(robust_lrs[layer], 0.0).int())
            flip_analysis[layer] = [n_flip, n_unflip]

        for i, id in enumerate(chosen_ids):
            client = clients[id]
            prop = client.n_sample / total_sample
            self.robust_lr_add_weights(original_params, robust_lrs, updates[i], prop)

        self.global_model.load_state_dict(original_params)
        return flip_analysis

    def compute_pairwise_distance(self, updates):
        def pairwise(u1, u2):
            ks = u1.keys()
            dist = 0
            for k in ks:
                if 'tracked' in k:
                    continue
                d = u1[k] - u2[k]
                dist = dist + torch.sum(d * d)
            return round(float(torch.sqrt(dist)), 2)

        scores = [0 for u in range(len(updates))]
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                dist = pairwise(updates[i], updates[j])
                scores[i] = scores[i] + dist
                scores[j] = scores[j] + dist
        return scores

    def bulyan_aggregate_global_model(self, clients: client_group, chosen_ids, pts):
        assert not clients is None and len(clients) > 0
        n_mal = 4
        original_params = self.global_model.state_dict()

        # collect client updates
        updates = list()
        for id in chosen_ids:
            client = clients[id]
            local_params = client.local_model.state_dict()
            update = OrderedDict()
            for layer, weight in local_params.items():
                update[layer] = local_params[layer] - original_params[layer]
            updates.append(update)

        temp_ids = list(copy.deepcopy(chosen_ids))

        krum_updates = list()
        n_ex = 2 * n_mal
        # print("Bulyan Stage 1：", len(updates))
        for i in range(len(chosen_ids)-n_ex):
            scores = self.compute_pairwise_distance(updates)
            n_update = len(updates)
            threshold = sorted(scores)[0]
            for k in range(n_update - 1, -1, -1):
                if scores[k] == threshold:
                    print("client {} is chosen:".format(temp_ids[k], round(scores[k], 2)))
                    krum_updates.append(updates[k])
                    del updates[k]
                    del temp_ids[k]
                    
        # print("Bulyan Stage 2：", len(krum_updates))    
        bulyan_update = OrderedDict()
        layers = krum_updates[0].keys()
        for layer in layers:
            bulyan_layer = None
            for update in krum_updates:
                bulyan_layer = update[layer][None, ...] if bulyan_layer is None else torch.cat(
                    (bulyan_layer, update[layer][None, ...]), 0)

            med, _ = torch.median(bulyan_layer, 0)
            _, idxs = torch.sort(torch.abs(bulyan_layer - med), 0)
            bulyan_layer = torch.gather(bulyan_layer, 0, idxs[:-n_ex, ...])
            # print("bulyan_layer",bulyan_layer.size())
            # bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            # print(bulyan_layer)
            if not 'tracked' in layer:
                bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            else:
                bulyan_update[layer] = torch.mean(bulyan_layer*1.0, 0).long()
            original_params[layer] = original_params[layer] + bulyan_update[layer]

        self.global_model.load_state_dict(original_params)
    
    def deepsight_aggregate_global_model(self, clients: client_group, chosen_ids, task, pts):
        def ensemble_cluster(neups, ddifs, biases):
            biases = np.array([bias.cpu().numpy() for bias in biases])
            #neups = np.array([neup.cpu().numpy() for neup in neups])
            #ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
            N = len(neups)
            # use bias to conduct DBSCAM
            # biases= np.array(biases)
            cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
            print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            print("neup_cluster:{}".format(neup_labels))
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            print("ddif_cluster:{}".format(ddif_labels))

            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]
                    
            print("dists_from_clusters:")
            print(dists_from_cluster)
            ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels
        
        global_weight = list(self.global_model.state_dict().values())[-2]
        global_bias = list(self.global_model.state_dict().values())[-1]

        biases = [(list(clients[i].local_model.state_dict().values())[-1] - global_bias) for i in chosen_ids]
        weights = [list(clients[i].local_model.state_dict().values())[-2] for i in chosen_ids]

        n_client = len(chosen_ids)
        cosine_similarity_dists = np.array((n_client, n_client))
        neups = list()
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in range(len(chosen_ids)):
            C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
            # print("C_nn:",C_nn)
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2
            
            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)
        # normalize
        neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
        print("n_exceeds:{}".format(n_exceeds))
        rand_input = None
        if isinstance(task, Cifar10FederatedTask):
            # 256 can be replaced with smaller value
            rand_input = torch.randn((256, 3, 32, 32)).to(self.device)
        elif isinstance(task, TinyImagenetFederatedTask):
            # 256 can be replaced with smaller value
            rand_input = torch.randn((256, 3, 64, 64)).to(self.device)

        global_ddif = torch.mean(torch.softmax(self.global_model(rand_input), dim=1), dim=0)
        # print("global_ddif:{} {}".format(global_ddif.size(),global_ddif))
        client_ddifs = [torch.mean(torch.softmax(clients[i].local_model(rand_input), dim=1), dim=0)/ global_ddif
                        for i in chosen_ids]
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
        # print("client_ddifs:{}".format(client_ddifs[0]))

        # use n_exceed to label
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        # print("deleted_clusters:",deleted_cluster_ids)
        temp_chosen_ids = copy.deepcopy(chosen_ids)
        for i in range(len(chosen_ids)-1, -1, -1):
            # print("cluster tag:",clusters[i])
            if clusters[i] in deleted_cluster_ids:
                del chosen_ids[i]

        print("final clients length:{}".format(len(chosen_ids)))
        if len(chosen_ids)==0:
            chosen_ids = temp_chosen_ids
        self.aggregate_global_model(clients, chosen_ids, None)
        
    def clip_weight_norm(self, clip=14):
        total_norm = self.get_global_model_norm()
        print("total_norm: " + str(total_norm) + "clip: " + str(clip))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in self.global_model.named_parameters():
                layer.data.mul_(clip_coef)
            current_norm = self.get_global_model_norm()
        return current_norm

    def add_differential_privacy_noise(self, sigma=0.001, cp=False):
        if not cp:
            for name, param in self.global_model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                # print(name)
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
        else:
            smoothed_model = copy.deepcopy(self.global_model)
            for name, param in smoothed_model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
            return smoothed_model

    def get_median_scores(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        median_counts = [0 for i in range(len(chosen_ids))]
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            median_count = task.get_median_counts(batch, clients, chosen_ids)
            median_counts = [x + y for x, y in zip(median_counts, median_count)]
        total_counts = sum(median_counts)
        normalized_median_counts = [(med_count / total_counts) for med_count in median_counts]
        return normalized_median_counts

    def get_avg_logits(self):
        pass
        # for i, data in enumerate(self.train_loader):
        #     # clear tddm
        #     batch = task.get_batch(i, data)

    def ensemble_distillation(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            batch = task.get_avg_logits(batch, clients, chosen_ids)
            self.optimizer.zero_grad()
            predicted_labels = self.global_model(batch.inputs)
            kl_div_loss = nn.KLDivLoss(reduction='batchmean')(predicted_labels.softmax(dim=-1).log(),
                                                              batch.labels.softmax(dim=-1))
            kl_div_loss.backward()
            self.optimizer.step()

    def adaptive_distillation(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            batch = task.get_median_logits(batch, clients, chosen_ids)
            self.optimizer.zero_grad()
            predicted_labels = self.global_model(batch.inputs)
            kl_div_loss = nn.KLDivLoss(reduction='batchmean')(predicted_labels.softmax(dim=-1).log(),
                                                              batch.labels.softmax(dim=-1))
            kl_div_loss.backward()
            self.optimizer.step()

    def fine_tuning(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            self.optimizer.zero_grad()
            predicted_labels = self.global_model(batch.inputs)
            # print("predicted_labels:",predicted_labels,batch.labels)
            loss = criterion(predicted_labels, batch.labels).mean()
            loss.backward()
            self.optimizer.step()"""
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