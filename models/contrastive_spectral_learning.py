#!/usr/bin/env python3
"""
Contrastive Spectral Learning
NT-Xent contrastive loss applied to spectral embeddings with dynamic wavelets
Self-supervised pre-training specifically designed for dynamic spectral features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import random

class SpectralPreservingAugmentation(nn.Module):
    """
    Graph augmentation that preserves spectral properties
    First augmentation method designed for spectral GAD
    """
    
    def __init__(self, 
                 preserve_eigenvalue_ratio: float = 0.8,
                 noise_level: float = 0.1,
                 edge_perturbation_ratio: float = 0.05):
        super().__init__()
        
        self.preserve_eigenvalue_ratio = preserve_eigenvalue_ratio
        self.noise_level = noise_level
        self.edge_perturbation_ratio = edge_perturbation_ratio
        
    def augment_graph(self, 
                     node_features: torch.Tensor,
                     eigenvalues: torch.Tensor,
                     eigenvectors: torch.Tensor,
                     edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Augment graph while preserving spectral structure
        """
        
        batch_size, num_nodes, num_features = node_features.shape
        
        augmented_features = self._add_spectral_preserving_noise(
            node_features, eigenvalues, eigenvectors
        )
        
        augmented_eigenvalues = self._perturb_eigenvalues(eigenvalues)
        
        augmented_eigenvectors = self._rotate_eigenvectors(eigenvectors)
        
        return augmented_features, augmented_eigenvalues, augmented_eigenvectors
    
    def _add_spectral_preserving_noise(self, 
                                      node_features: torch.Tensor,
                                      eigenvalues: torch.Tensor, 
                                      eigenvectors: torch.Tensor) -> torch.Tensor:
        """
        Add noise that preserves spectral energy distribution
        """
        
        batch_size, num_nodes, num_features = node_features.shape
        
        # Transform to spectral domain
        spectral_features = torch.matmul(eigenvectors.transpose(-2, -1), node_features)
        
        # Add noise proportional to eigenvalue importance
        eigenvalue_weights = eigenvalues / (eigenvalues.sum(dim=-1, keepdim=True) + 1e-8)
        noise_scale = self.noise_level * eigenvalue_weights.unsqueeze(-1)
        
        spectral_noise = torch.randn_like(spectral_features) * noise_scale
        augmented_spectral = spectral_features + spectral_noise
        
        # Transform back to spatial domain
        augmented_features = torch.matmul(eigenvectors, augmented_spectral)
        
        return augmented_features
    
    def _perturb_eigenvalues(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        perturbation = torch.randn_like(eigenvalues) * self.noise_level * 0.1
        perturbed_eigenvalues = eigenvalues + perturbation
        
        # Maintain non-negative and sorted order
        perturbed_eigenvalues = torch.clamp(perturbed_eigenvalues, min=0.001)
        perturbed_eigenvalues, _ = torch.sort(perturbed_eigenvalues, dim=-1)
        
        return perturbed_eigenvalues
    
    def _rotate_eigenvectors(self, eigenvectors: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, num_eigenvals = eigenvectors.shape
        rotation_angles = torch.randn(batch_size, num_eigenvals) * self.noise_level * 0.05
        rotation_noise = torch.randn_like(eigenvectors) * rotation_angles.unsqueeze(1)
        rotated_eigenvectors = eigenvectors + rotation_noise
        Q, R = torch.linalg.qr(rotated_eigenvectors)
        
        return Q
    
    def create_augmentation_pairs(self,
                                 node_features: torch.Tensor,
                                 eigenvalues: torch.Tensor,
                                 eigenvectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create positive pairs for contrastive learning
        """
        
        # first augmentation
        aug1_features, aug1_eigenvals, aug1_eigenvecs = self.augment_graph(
            node_features, eigenvalues, eigenvectors
        )
        
        # second augmentation
        aug2_features, aug2_eigenvals, aug2_eigenvecs = self.augment_graph(
            node_features, eigenvalues, eigenvectors
        )
        
        return {
            'original': (node_features, eigenvalues, eigenvectors),
            'augmentation_1': (aug1_features, aug1_eigenvals, aug1_eigenvecs),
            'augmentation_2': (aug2_features, aug2_eigenvals, aug2_eigenvecs)
        }

class SpectralContrastiveLoss(nn.Module):
    """
    Contrastive loss for spectral embeddings with NT-Xent loss adapted for dynamic spectral representations
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 spectral_weight: float = 0.5,
                 temporal_weight: float = 0.3,
                 spatial_weight: float = 0.2):
        super().__init__()
        
        self.temperature = temperature
        self.spectral_weight = spectral_weight
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        
        total_weight = spectral_weight + temporal_weight + spatial_weight
        self.spectral_weight /= total_weight
        self.temporal_weight /= total_weight
        self.spatial_weight /= total_weight
    
    def compute_spectral_similarity(self, 
                                   embeddings1: torch.Tensor,
                                   embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity in spectral embedding space
        """
        
        # L2 normalize embeddings
        embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
        embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
        
        # Cosine similarity
        similarity = torch.matmul(embeddings1_norm, embeddings2_norm.transpose(-2, -1))
        
        return similarity
    
    def nt_xent_loss(self, 
                     embeddings1: torch.Tensor,
                     embeddings2: torch.Tensor) -> torch.Tensor:
        """
        normalized temperature scaled cross-entropy Loss for contrastive learning
        """
        
        batch_size = embeddings1.shape[0]
        
        # Concatenate embeddings
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)  
        
        # Compute similarity matrix
        similarity_matrix = self.compute_spectral_similarity(embeddings, embeddings)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size).to(embeddings.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(embeddings.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def spectral_consistency_loss(self,
                                 spectral_features1: torch.Tensor,
                                 spectral_features2: torch.Tensor,
                                 eigenvalues1: torch.Tensor,
                                 eigenvalues2: torch.Tensor) -> torch.Tensor:
        """
        Spectral consistency loss for preserved spectral properties
        """
        
        # energy distribution 
        energy1 = torch.sum(spectral_features1 ** 2, dim=-1)
        energy2 = torch.sum(spectral_features2 ** 2, dim=-1)
        
        energy_dist1 = F.softmax(energy1, dim=-1)
        energy_dist2 = F.softmax(energy2, dim=-1)
        
        # KL divergence between energy distributions
        energy_consistency = F.kl_div(
            torch.log(energy_dist1 + 1e-8), energy_dist2, reduction='batchmean'
        )
        
        # eigenvalue consistency
        eigenval_consistency = F.mse_loss(eigenvalues1, eigenvalues2)
        
        return energy_consistency + 0.1 * eigenval_consistency
    
    def forward(self,
               original_embeddings: torch.Tensor,
               aug1_embeddings: torch.Tensor, 
               aug2_embeddings: torch.Tensor,
               original_spectral: Optional[torch.Tensor] = None,
               aug1_spectral: Optional[torch.Tensor] = None,
               aug2_spectral: Optional[torch.Tensor] = None,
               original_eigenvals: Optional[torch.Tensor] = None,
               aug1_eigenvals: Optional[torch.Tensor] = None,
               aug2_eigenvals: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component contrastive loss
        """
        
        # contrastive loss (original vs augmentation)
        contrastive_loss_1 = self.nt_xent_loss(original_embeddings, aug1_embeddings)
        contrastive_loss_2 = self.nt_xent_loss(original_embeddings, aug2_embeddings)
        contrastive_loss_3 = self.nt_xent_loss(aug1_embeddings, aug2_embeddings)
        
        spatial_contrastive = (contrastive_loss_1 + contrastive_loss_2 + contrastive_loss_3) / 3.0
        
        # spectral consistency loss
        spectral_consistency = torch.tensor(0.0)
        if all(x is not None for x in [original_spectral, aug1_spectral, aug2_spectral]):
            spec_loss_1 = self.spectral_consistency_loss(
                original_spectral, aug1_spectral, original_eigenvals, aug1_eigenvals
            )
            spec_loss_2 = self.spectral_consistency_loss(
                original_spectral, aug2_spectral, original_eigenvals, aug2_eigenvals
            )
            spectral_consistency = (spec_loss_1 + spec_loss_2) / 2.0
        
        # combined loss
        total_loss = (
            self.spatial_weight * spatial_contrastive +
            self.spectral_weight * spectral_consistency
        )
        
        return {
            'total_loss': total_loss,
            'spatial_contrastive': spatial_contrastive,
            'spectral_consistency': spectral_consistency
        }

class ContrastiveSpectralPretrainer(nn.Module):
    """
    self-supervised pre-training for spectral GAD with contrastive learning specifically designed for dynamic spectral features
    """
    
    def __init__(self,
                 spectral_model: nn.Module,
                 projection_dim: int = 128,
                 temperature: float = 0.1):
        super().__init__()
        
        self.spectral_model = spectral_model
        self.projection_dim = projection_dim
        
        # projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(spectral_model.hidden_dim, spectral_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(spectral_model.hidden_dim, projection_dim)
        )
        
        # augmentation module
        self.augmentation = SpectralPreservingAugmentation()
        
        # contrastive loss
        self.contrastive_loss = SpectralContrastiveLoss(temperature=temperature)
    
    def forward(self, 
               temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        contrastive pre-training forward pass
        """

        node_features, eigenvalues, eigenvectors = temporal_graph_sequence[-1]
        
        # augmentation pairs
        augmentation_pairs = self.augmentation.create_augmentation_pairs(
            node_features, eigenvalues, eigenvectors
        )
        
        # original and augmented versions
        embeddings = {}
        spectral_features = {}
        
        for key, (features, eigenvals, eigenvecs) in augmentation_pairs.items():
            # create temporal sequence for this augmentation
            aug_sequence = temporal_graph_sequence[:-1] + [(features, eigenvals, eigenvecs)]
            
            # spectral embeddings
            with torch.no_grad() if key != 'original' else torch.enable_grad():
                results = self.spectral_model(aug_sequence)
                embedding = results['spectral_features'].mean(dim=1)  # Average over nodes
                
            # project to contrastive space
            projected = self.projection_head(embedding)
            embeddings[key] = projected
            
            # store spectral features
            spectral_features[key] = results.get('energy_distributions', None)
        
        # compute contrastive loss
        loss_components = self.contrastive_loss(
            embeddings['original'],
            embeddings['augmentation_1'],
            embeddings['augmentation_2'],
            spectral_features['original'],
            spectral_features['augmentation_1'],
            spectral_features['augmentation_2'],
            augmentation_pairs['original'][1],  # eigenvalues
            augmentation_pairs['augmentation_1'][1],
            augmentation_pairs['augmentation_2'][1]
        )
        
        return {
            'embeddings': embeddings,
            'loss_components': loss_components,
            'augmentation_pairs': augmentation_pairs
        }
    
    def pretrain_step(self, 
                     temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single pre-training step
        """
        
        optimizer.zero_grad()
        results = self.forward(temporal_graph_sequence)
        loss = results['loss_components']['total_loss']
        loss.backward()
        optimizer.step()
        
        # log losses 
        return {
            'total_loss': loss.item(),
            'spatial_contrastive': results['loss_components']['spatial_contrastive'].item(),
            'spectral_consistency': results['loss_components']['spectral_consistency'].item()
        }

class ContrastiveSpectralTrainer:
    """
    Training for contrastive spectral pre-training
    """
    
    def __init__(self, 
                 model: ContrastiveSpectralPretrainer,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.training_history = {
            'total_loss': [],
            'spatial_contrastive': [],
            'spectral_consistency': []
        }
    
    def pretrain(self,
                train_loader: torch.utils.data.DataLoader,
                num_epochs: int = 100,
                log_interval: int = 10) -> Dict[str, List[float]]:
        """
        Run contrastive pre-training
        """
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = {'total_loss': [], 'spatial_contrastive': [], 'spectral_consistency': []}
            
            for batch_idx, temporal_sequence in enumerate(train_loader):
                losses = self.model.pretrain_step(temporal_sequence, self.optimizer)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            for key in epoch_losses:
                avg_loss = np.mean(epoch_losses[key])
                self.training_history[key].append(avg_loss)
            
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}/{num_epochs}:")
                for key, avg_loss in zip(epoch_losses.keys(), [np.mean(epoch_losses[k]) for k in epoch_losses.keys()]):
                    print(f"  {key}: {avg_loss:.4f}")
        
        return self.training_history
    
    def get_pretrained_encoder(self) -> nn.Module:
        """
        Extract pre-trained spectral encoder
        """
        return self.model.spectral_model

def create_contrastive_pretrainer(spectral_model: nn.Module, config: Dict) -> ContrastiveSpectralPretrainer:
    """
    function to create contrastive pre-trainer
    """
    
    pretrainer = ContrastiveSpectralPretrainer(
        spectral_model=spectral_model,
        projection_dim=config.get('projection_dim', 128),
        temperature=config.get('temperature', 0.1)
    )
    
    return pretrainer

if __name__ == "__main__":
    print("Contrastive Spectral Learning")
    print("=" * 40)
    
    # import base model
    from temporal_spectral_fusion import create_temporal_spectral_model
    
    # create base spectral model
    model_config = {
        'input_dim': 64,
        'hidden_dim': 128,
        'temporal_steps': 5,
        'domain_type': 'financial'
    }
    
    base_model = create_temporal_spectral_model(model_config)
    
    # create contrastive pre-trainer
    pretrainer_config = {
        'projection_dim': 128,
        'temperature': 0.1
    }
    
    pretrainer = create_contrastive_pretrainer(base_model, pretrainer_config)
    
    batch_size, num_nodes, num_features = 4, 100, 64
    temporal_sequence = []
    
    for t in range(3):
        node_features = torch.randn(batch_size, num_nodes, num_features)
        eigenvalues = torch.sort(torch.rand(batch_size, 20))[0]
        eigenvectors = torch.randn(batch_size, num_nodes, 20)
        temporal_sequence.append((node_features, eigenvalues, eigenvectors))
    
    trainer = ContrastiveSpectralTrainer(pretrainer)
    optimizer = trainer.optimizer
    
    losses = pretrainer.pretrain_step(temporal_sequence, optimizer)
    
    print("Pre-training losses:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")
    