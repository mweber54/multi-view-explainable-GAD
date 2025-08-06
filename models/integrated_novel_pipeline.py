#!/usr/bin/env python3
"""
Integrated Novel Pipeline
Combines all original contributions into a unified architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path

from temporal_spectral_fusion import TemporalSpectralFusion, create_temporal_spectral_model
from explainable_spectral_analysis import MultiViewSpectralExplainer, create_explainable_spectral_model
from contrastive_spectral_learning import ContrastiveSpectralPretrainer, create_contrastive_pretrainer

class NovelSpectralGADPipeline(nn.Module):
    """
    Complete Novel Architecture integrating all our original contributions:
    
    1. Temporal-Spectral Fusion (dynamic wavelets + temporal modeling)
    2. Energy-Guided Multi-Scale Analysis 
    3. Contrastive Spectral Pre-training
    4. Multi-View Explainable Analysis
    5. Domain-Adaptive Components
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.mode = config.get('mode', 'full')  
        self.spectral_model = create_temporal_spectral_model(config)
        
        # Contrastive pre-trainer for self-supervised learning
        if self.mode in ['pretrain', 'full']:
            self.contrastive_pretrainer = create_contrastive_pretrainer(
                self.spectral_model, config.get('contrastive', {})
            )
        
        # explainable module
        if self.mode in ['finetune', 'full']:
            self.explainer = create_explainable_spectral_model(
                self.spectral_model, config.get('explainer', {})
            )
        
        self.ensemble_components = self._create_ensemble_components(config)
        self.final_classifier = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 128), config.get('hidden_dim', 128) // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.get('hidden_dim', 128) // 2, 1),
            nn.Sigmoid()
        )
    
    def _create_ensemble_components(self, config: Dict) -> nn.ModuleDict:
        """Create ensemble components for prediction"""
        
        hidden_dim = config.get('hidden_dim', 128)

        components = nn.ModuleDict({
            'energy_scorer': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            
            'distance_scorer': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            
            'spectral_scorer': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            
            'ensemble_combiner': nn.Sequential(
                nn.Linear(3, 8),  
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        })
        
        return components
    
    def pretrain_mode(self, temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Self-supervised pre-training mode using contrastive learning
        """
        
        if not hasattr(self, 'contrastive_pretrainer'):
            raise ValueError("Contrastive pretrainer not available in current mode")
        
        contrastive_results = self.contrastive_pretrainer(temporal_graph_sequence)

        return {
            'mode': 'pretrain',
            'contrastive_loss': contrastive_results['loss_components']['total_loss'],
            'embeddings': contrastive_results['embeddings']['original'],
            'augmentation_info': contrastive_results['augmentation_pairs'].keys()
        }
    
    def finetune_mode(self, 
                     temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     labels: Optional[torch.Tensor] = None,
                     explain: bool = False) -> Dict[str, torch.Tensor]:
        """
        Supervised fine-tuning mode with optional explanations
        """
        
        spectral_results = self.spectral_model(temporal_graph_sequence)
        spectral_features = spectral_results['spectral_features']
        batch_size, num_nodes, feature_dim = spectral_features.shape
        ensemble_scores = self._compute_ensemble_scores(spectral_features)
        final_scores = self.final_classifier(spectral_features.mean(dim=1))  # Average over nodes
        
        results = {
            'mode': 'finetune',
            'anomaly_scores': spectral_results['anomaly_scores'],
            'ensemble_scores': ensemble_scores,
            'final_scores': final_scores.squeeze(),
            'spectral_features': spectral_features,
            'energy_distributions': spectral_results.get('energy_distributions', None)
        }
        
        # add explanations if requested
        if explain and hasattr(self, 'explainer'):
            explanations = self.explainer.explain_prediction(temporal_graph_sequence)
            results['explanations'] = explanations
        
        # compute loss if labels provided
        if labels is not None:
            supervised_loss = F.binary_cross_entropy(
                results['final_scores'], labels.float()
            )
            results['supervised_loss'] = supervised_loss
        
        return results
    
    def full_mode(self, 
                 temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 labels: Optional[torch.Tensor] = None,
                 explain: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full pipeline mode combining all components
        """
        
        # pre-training 
        contrastive_results = self.pretrain_mode(temporal_graph_sequence)
        
        # fine-tuning 
        finetune_results = self.finetune_mode(temporal_graph_sequence, labels, explain)
        
        # combine results
        combined_scores = 0.7 * finetune_results['final_scores'] + 0.3 * contrastive_results['embeddings'].mean(dim=1)
        
        results = {
            'mode': 'full',
            'combined_anomaly_scores': combined_scores,
            'contrastive_component': contrastive_results,
            'supervised_component': finetune_results,
            'final_prediction': combined_scores
        }
        
        return results
    
    def _compute_ensemble_scores(self, spectral_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute ensemble component scores"""
        
        aggregated_features = spectral_features.mean(dim=1)  # [batch, features]
        energy_scores = self.ensemble_components['energy_scorer'](aggregated_features)
        distance_scores = self.ensemble_components['distance_scorer'](aggregated_features)
        spectral_scores = self.ensemble_components['spectral_scorer'](aggregated_features)
        component_scores = torch.cat([energy_scores, distance_scores, spectral_scores], dim=-1)
        ensemble_score = self.ensemble_components['ensemble_combiner'](component_scores)
        
        return {
            'energy_scores': energy_scores.squeeze(),
            'distance_scores': distance_scores.squeeze(),
            'spectral_scores': spectral_scores.squeeze(),
            'ensemble_score': ensemble_score.squeeze()
        }
    
    def forward(self, 
               temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
               labels: Optional[torch.Tensor] = None,
               explain: bool = False) -> Dict[str, torch.Tensor]:
        """
        Main forward pass
        """
        
        if self.mode == 'pretrain':
            return self.pretrain_mode(temporal_graph_sequence)
        elif self.mode == 'finetune':
            return self.finetune_mode(temporal_graph_sequence, labels, explain)
        else:  # 'full'
            return self.full_mode(temporal_graph_sequence, labels, explain)

class NovelPipelineTrainer:
    """
    Training orchestrator for the complete novel pipeline
    """
    
    def __init__(self, 
                 model: NovelSpectralGADPipeline,
                 config: Dict):
        
        self.model = model
        self.config = config
        
        # optimizers for different phases
        self.pretrain_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('pretrain_lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.finetune_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('finetune_lr', 5e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # training history
        self.training_history = {
            'pretrain': {'contrastive_loss': []},
            'finetune': {'supervised_loss': [], 'accuracy': []}
        }
    
    def pretrain_phase(self, 
                      train_loader: torch.utils.data.DataLoader,
                      num_epochs: int = 50) -> Dict[str, List[float]]:
        """
        Self-supervised pre-training phase
        """
        self.model.mode = 'pretrain'
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, temporal_sequence in enumerate(train_loader):
                self.pretrain_optimizer.zero_grad()
                results = self.model(temporal_sequence)
                loss = results['contrastive_loss']
                loss.backward()
                self.pretrain_optimizer.step()
                epoch_losses.append(loss.item())
            
            # log epochs
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.training_history['pretrain']['contrastive_loss'].append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Contrastive Loss: {avg_loss:.4f}")
        
        print("Pre-training Phase Complete!")
        return self.training_history['pretrain']
    
    def finetune_phase(self,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Supervised fine-tuning phase
        """
        
        self.model.mode = 'finetune'
        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            
            for batch_idx, (temporal_sequence, labels) in enumerate(train_loader):
                self.finetune_optimizer.zero_grad()
                results = self.model(temporal_sequence, labels)
                loss = results['supervised_loss']
                loss.backward()
                self.finetune_optimizer.step()
                train_losses.append(loss.item())
            
            val_accuracy = self._evaluate(val_loader)
            
            # log progress
            avg_train_loss = sum(train_losses) / len(train_losses)
            self.training_history['finetune']['supervised_loss'].append(avg_train_loss)
            self.training_history['finetune']['accuracy'].append(val_accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        return self.training_history['finetune']
    
    def _evaluate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate model on validation set"""
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for temporal_sequence, labels in val_loader:
                results = self.model(temporal_sequence)
                predictions = (results['final_scores'] > 0.5).float()
                
                correct += (predictions == labels.float()).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def full_training_pipeline(self,
                              pretrain_loader: torch.utils.data.DataLoader,
                              train_loader: torch.utils.data.DataLoader,
                              val_loader: torch.utils.data.DataLoader,
                              pretrain_epochs: int = 50,
                              finetune_epochs: int = 100) -> Dict:
        """
        Complete training pipeline
        """
        # Phase 1: Self-supervised pre-training
        pretrain_history = self.pretrain_phase(pretrain_loader, pretrain_epochs)
        
        # Phase 2: Supervised fine-tuning
        finetune_history = self.finetune_phase(train_loader, val_loader, finetune_epochs)
        
        # Switch to full mode for final evaluation
        self.model.mode = 'full'
        
        return {
            'pretrain_history': pretrain_history,
            'finetune_history': finetune_history,
            'final_model_mode': 'full'
        }

def create_novel_pipeline(config: Dict) -> NovelSpectralGADPipeline:
    """
    Factory function to create complete novel pipeline
    """
    
    required_keys = ['input_dim', 'hidden_dim', 'temporal_steps', 'domain_type']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    pipeline = NovelSpectralGADPipeline(config)
    
    return pipeline

def create_example_config() -> Dict:
    """Create example configuration for the novel pipeline"""
    
    config = {
        # Core model parameters
        'input_dim': 64,
        'hidden_dim': 128,
        'temporal_steps': 10,
        'domain_type': 'financial',  # 'financial', 'social', 'general'
        'mode': 'full',  # 'pretrain', 'finetune', 'full'
        
        # Contrastive learning parameters
        'contrastive': {
            'projection_dim': 128,
            'temperature': 0.1
        },
        
        # Explainer parameters
        'explainer': {
            'num_wavelets': 8,
            'consistency_threshold': 0.7
        },
        
        # Training parameters
        'pretrain_lr': 1e-3,
        'finetune_lr': 5e-4,
        'weight_decay': 1e-5,
        
        # Model features
        'use_energy_tracking': True,
        'use_multi_scale': True,
        'use_domain_adaptation': True
    }
    
    return config

if __name__ == "__main__":
    # Demonstration of the complete novel pipeline
    print("Novel Spectral GAD Pipeline - Complete Integration")
    print("=" * 60)
    
    # create configuration
    config = create_example_config()
    print("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    print("\nCreating Novel Pipeline...")
    pipeline = create_novel_pipeline(config)
    
    # Example temporal graph sequence
    batch_size, num_nodes, num_features = 4, 100, 64
    temporal_sequence = []
    
    for t in range(config['temporal_steps']):
        node_features = torch.randn(batch_size, num_nodes, num_features)
        eigenvalues = torch.sort(torch.rand(batch_size, 20))[0]
        eigenvectors = torch.randn(batch_size, num_nodes, 20)
        temporal_sequence.append((node_features, eigenvalues, eigenvectors))
    
    labels = torch.randint(0, 2, (batch_size,))
    print("\nTesting Pipeline Modes:")
    
    # 1. Pre-training 
    pipeline.mode = 'pretrain'
    print("1. Pre-training Mode:")
    with torch.no_grad():
        pretrain_results = pipeline(temporal_sequence)
        print(f"   Contrastive loss: {pretrain_results['contrastive_loss']:.4f}")
    
    # 2. Fine-tuning 
    pipeline.mode = 'finetune'
    print("2. Fine-tuning Mode:")
    with torch.no_grad():
        finetune_results = pipeline(temporal_sequence, labels, explain=True)
        print(f"   Final scores shape: {finetune_results['final_scores'].shape}")
        print(f"   Has explanations: {'explanations' in finetune_results}")
    
    # 3. Full 
    pipeline.mode = 'full'
    print("3. Full Mode:")
    with torch.no_grad():
        full_results = pipeline(temporal_sequence, labels, explain=True)
        print(f"   Combined scores shape: {full_results['combined_anomaly_scores'].shape}")
        print(f"   Has contrastive component: {'contrastive_component' in full_results}")
        print(f"   Has supervised component: {'supervised_component' in full_results}")
    config_path = Path("experiments/novel_pipeline_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")