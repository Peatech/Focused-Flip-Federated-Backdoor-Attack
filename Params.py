from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import logging
import torch

ALL_TASKS = ['backdoor', 'normal', 'sentinet_evasion',  # 'spectral_evasion',
             'neural_cleanse', 'mask_norm', 'sums', 'neural_cleanse_part1']

posion_image_ids = []

@dataclass
class Params:
    '''defence rules'''
    defence: str = 'fedavg' # mediod-distillation, ensemble-distillation, robustlr, finetuning , certified-robustness, fedavgcka
    '''task and model'''
    task: str = 'CifarFed' #CifarFed
    model: str = 'resnet18' #resnet18
    pretrained: str = True

    '''device'''
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''dataset'''
    transform_train = True

    '''epochs and batchsize'''
    test_batch_size: int = 32
    batch_size: int = 64
    local_epoch: int =2
    n_epochs: int = 100

    max_batch_id: int = None
    input_shape = None

    '''loss and optimizer'''
    loss_tasks: List = field(default_factory=lambda: ['noraml', 'backdoor'])
    loss_balance: str = 'fixed'
    optimizer: str = 'SGD'

    '''learning rate and weight decay'''
    lr: int = 0.001
    decay: float = 0.0001
    momentum: float = 0.9
    '''clients'''
    n_clients: int = 20
    #select
    n_malicious_client: int = 4
    chosen_rate: float = 0.5

    '''dataset path'''
    data_path: str = '.data/'
    save_model: bool = True

    '''synthesizer'''
    synthesizer: str = 'pattern'
    backdoor_dynamic_position: bool = False

    # relabel images with poison_number
    poison_images: List[int] = field(default_factory=lambda: [])
    poison_images_test: List[int] = None
    backdoor_label: int = 8

    # nc evasion
    nc_p_norm: int = 1

    poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
    mgda_normalize: str = None

    clip_batch: float = None
    running_scales = None
    
    #flip_rate
    conv_rate: float = 0.02
    fc_rate: float = 0.001

    fixed_scales: Dict[str, float] = field(default_factory=lambda: {'normal':0.6,'backdoor':0.4})
    # fixed_mal =True
    
    # Temporary storage for running values
    running_losses = None

    # FL params
    fl_weight_scale: int = 1
    fl_local_epoch = 1
    

    # handcrafted trigger
    handcraft = True
    handcraft_trigger = True
    distributed_trigger = False
    
    
    acc_threshold = 0.01

    # file path
    resume_model: str = None

    # freeze
    freezing = False
    
    # for flip
    flip_factor = 1
    
    # if attack bulyan, should set a number>0, for example 0.7
    model_similarity_factor: float = 0.0
    
    norm_clip_factor: float = 10.0
    
    heterogenuity:float = 1.0
    
    

    # differencial privacy
    dp: bool = False

    kernel_selection: str = "movement"

#     attention_visualization: bool = False
    
    #server_dataset
    server_dataset = False
    resultdir = 'result-fedavg'

    # FedAvgCKA Defense Configuration
    fedavgcka_enabled: bool = False
    "Enable FedAvgCKA pre-aggregation defense"

    fedavgcka_root_dataset_size: int = 64
    "Size of root dataset R for activation extraction (paper shows 16+ effective)"

    fedavgcka_root_dataset_strategy: str = "class_balanced"
    "Strategy for root dataset sampling: 'random' or 'class_balanced'"

    fedavgcka_layer_comparison: str = "multi_layer"
    "Layer(s) for CKA comparison: 'penultimate', 'layer3', 'layer2', or 'multi_layer'"

    fedavgcka_trim_fraction: float = 0.5
    "Fraction of clients to exclude based on CKA scores (0.5 = exclude bottom 50%)"

    fedavgcka_multi_layer_weights: Dict[str, float] = field(default_factory=lambda: {
        'penultimate': 0.6,
        'layer3': 0.3, 
        'layer2': 0.1
    })
    "Weights for combining CKA scores across multiple layers (when using multi_layer mode)"

    fedavgcka_numerical_eps: float = 1e-12
    "Small constant for numerical stability in CKA computation"

    fedavgcka_log_scores: bool = True
    "Whether to log detailed CKA scores and client selections"

    # FedSPECTRE-Hybrid Defense Configuration
    fedspectre_enabled: bool = False
    "Enable FedSPECTRE-Hybrid pre-aggregation defense"

    fedspectre_rank: int = 128
    "Rank for low-rank projection in robust statistics (default 128)"

    fedspectre_alpha: float = 0.4
    "Weight for CKA component in anomaly score (default 0.4)"

    fedspectre_beta: float = 0.3
    "Weight for augmentation stability component (default 0.3)"

    fedspectre_gamma: float = 0.3
    "Weight for spectral projection component (default 0.3)"

    fedspectre_trim_fraction: float = 0.5
    "Fraction of clients to exclude based on anomaly scores (default 0.5)"

    fedspectre_trim_fraction_cov: float = 0.05
    "Fraction to trim before covariance estimation (default 0.05)"

    fedspectre_log_scores: bool = True
    "Whether to log detailed anomaly scores and client selections"

    def __post_init__(self):
        # enable logging anyways when saving statistics
        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)
        
        # Auto-enable defenses based on defence parameter (ensure only one is enabled)
        if self.defence == 'fedavgcka':
            self.fedavgcka_enabled = True
            self.fedspectre_enabled = False
        elif self.defence == 'fedspectre':
            self.fedspectre_enabled = True
            self.fedavgcka_enabled = False
        else:
            # For other defenses, disable both
            self.fedavgcka_enabled = False
            self.fedspectre_enabled = False

