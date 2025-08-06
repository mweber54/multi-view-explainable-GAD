import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    features_path: str = "node_features.csv"
    edges_path: str = "edges.csv"
    labels_path: str = "node_labels.csv"
    train_mask_path: str = "train_mask.csv"
    val_mask_path: str = "val_mask.csv"
    test_mask_path: str = "test_mask.csv"
    static_save_path: str = "amazon_static.pt"
    embeddings_path: str = "embeddings.npy"
    
    # feature engineering
    standardize_features: bool = True
    include_centrality: bool = True
    include_motifs: bool = True
    include_temporal: bool = True
    betweenness_sample_k: int = 100
    
    # label mapping
    label_mapping: dict = None
    
    def __post_init__(self):
        if self.label_mapping is None:
            self.label_mapping = {
                0: 0, 1: 1,
                '0': 0, '1': 1,
                'unknown': -1, -1: -1
            }


@dataclass 
class ModelConfig:
    """Configuration for model architecture"""
    # encoder architecture
    hidden_channels: int = 128
    latent_dim: int = 64
    projection_dim: int = 64
    dropout: float = 0.3
    num_heads: int = 2
    
    # classifier
    classifier_hidden: int = 64


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # pretraining
    epochs_pretrain: int = 35
    epochs_finetune: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # contrastive learning
    temperature: float = 0.5
    edge_dropout_prob: float = 0.2
    feature_mask_prob: float = 0.2
    
    # batch processing
    pretrain_batch_size: int = 1024
    finetune_batch_size: int = 2048
    eval_batch_size: int = 2048
    num_neighbors: List[int] = None
    
    # optimization
    gradient_clip_norm: float = 1.0
    
    # checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    
    def __post_init__(self):
        if self.num_neighbors is None:
            self.num_neighbors = [10, 10]


@dataclass
class SystemConfig:
    """Configuration for system settings"""
    device: str = "auto"  # "auto", "cpu", "cuda"
    random_seed: int = 42
    num_workers: int = 0
    
    # logging
    log_level: str = "INFO"
    log_file: Optional[str] = "gad_pipeline.log"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig = None
    model: ModelConfig = None  
    training: TrainingConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary"""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # validate file paths exist
        required_files = [
            self.data.features_path,
            self.data.edges_path, 
            self.data.labels_path
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                errors.append(f"Required file not found: {file_path}")
        
        # validate ranges
        if self.model.dropout < 0 or self.model.dropout > 1:
            errors.append("Dropout must be between 0 and 1")
            
        if self.training.temperature <= 0:
            errors.append("Temperature must be positive")
            
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        return errors


# default configuration instance
default_config = Config()