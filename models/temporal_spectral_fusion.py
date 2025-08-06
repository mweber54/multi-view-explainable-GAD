#!/usr/bin/env python3
"""
Temporal-Spectral Fusion Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class DynamicWaveletModule(nn.Module):
    """
    Domain-adaptive dynamic wavelets
    """
    
    def __init__(self, 
                 num_wavelets: int = 8,
                 max_frequency: float = 2.0,
                 domain_type: str = 'general',
                 learnable: bool = True):
        super().__init__()
        
        self.num_wavelets = num_wavelets
        self.max_frequency = max_frequency
        self.domain_type = domain_type
        self.learnable = learnable
        
        # domain-adaptive initialization 
        if learnable:
            self.wavelet_centers = nn.Parameter(self._init_domain_centers())
            self.wavelet_widths = nn.Parameter(self._init_domain_widths())
            self.wavelet_amplitudes = nn.Parameter(torch.ones(num_wavelets))
            self.domain_constraints = self._create_domain_constraints()
        else:
            # fixed wavelets for comparison
            self.register_buffer('wavelet_centers', self._init_domain_centers())
            self.register_buffer('wavelet_widths', self._init_domain_widths())
            self.register_buffer('wavelet_amplitudes', torch.ones(num_wavelets))
    
    def _init_domain_centers(self) -> torch.Tensor:
        """
        Domain-adaptive wavelet center initialization
        Financial graphs: Focus on transaction burst frequencies
        Social graphs: Focus on community structure frequencies
        Temporal graphs: Balanced across temporal frequencies
        """
        
        if self.domain_type == 'financial':
            centers = torch.tensor([0.1, 0.3, 0.6, 1.0, 1.4, 1.8, 2.2, 2.5])[:self.num_wavelets]
        elif self.domain_type == 'social':
            centers = torch.tensor([0.05, 0.15, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])[:self.num_wavelets]
        elif self.domain_type == 'temporal':
            centers = torch.linspace(0.1, self.max_frequency, self.num_wavelets)
        else:
            # uniform distribution
            centers = torch.linspace(0.1, self.max_frequency, self.num_wavelets)
        
        return centers.float()
    
    def _init_domain_widths(self) -> torch.Tensor:
        """Domain-adaptive wavelet width initialization"""
        
        if self.domain_type == 'financial':
            widths = torch.full((self.num_wavelets,), 0.05)
        elif self.domain_type == 'social':
            widths = torch.full((self.num_wavelets,), 0.15)
        else:
            widths = torch.full((self.num_wavelets,), 0.1)
        
        return widths.float()
    
    def _create_domain_constraints(self) -> Dict:
        """Create domain-specific parameter constraints"""
        
        constraints = {
            'center_min': 0.01,
            'center_max': self.max_frequency,
            'width_min': 0.01,
            'width_max': 0.5,
            'amplitude_min': 0.1,
            'amplitude_max': 2.0
        }
        
        if self.domain_type == 'financial':
            constraints['width_max'] = 0.2  # keep wavelets sharp for transactions
        elif self.domain_type == 'social':
            constraints['width_min'] = 0.05  
        
        return constraints
    
    def _apply_constraints(self):
        """Apply domain-specific constraints to parameters"""
        if not self.learnable:
            return
            
        with torch.no_grad():
            self.wavelet_centers.clamp_(
                self.domain_constraints['center_min'],
                self.domain_constraints['center_max']
            )
            
            self.wavelet_widths.clamp_(
                self.domain_constraints['width_min'],
                self.domain_constraints['width_max']
            )

            self.wavelet_amplitudes.clamp_(
                self.domain_constraints['amplitude_min'],
                self.domain_constraints['amplitude_max']
            )
    
    def generate_wavelets(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Generate dynamic wavelets based on graph eigenspectrum and adapt to actual eigenvalue distribution
        """
        
        self._apply_constraints()
        
        # wavelets for each eigenvalue
        batch_size = eigenvalues.shape[0] if eigenvalues.dim() > 1 else 1
        num_eigenvals = eigenvalues.shape[-1]
        eigenvals = eigenvalues.view(-1, num_eigenvals, 1)  
        centers = self.wavelet_centers.view(1, 1, self.num_wavelets)  
        widths = self.wavelet_widths.view(1, 1, self.num_wavelets)
        amplitudes = self.wavelet_amplitudes.view(1, 1, self.num_wavelets)
        # gaussian wavelets 
        wavelet_responses = amplitudes * torch.exp(
            -0.5 * ((eigenvals - centers) / widths) ** 2
        )
        
        return wavelet_responses  # [batch, eigenvals, wavelets]
    
    def forward(self, graph_signal: torch.Tensor, eigenvalues: torch.Tensor, 
                eigenvectors: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic wavelets to graph signal
        """
        
        wavelets = self.generate_wavelets(eigenvalues)  # [batch, eigenvals, wavelets]
        spectral_signal = torch.matmul(eigenvectors.transpose(-2, -1), graph_signal)
        filtered_signals = []
        for i in range(self.num_wavelets):
            wavelet_filter = wavelets[:, :, i].unsqueeze(-1)  # [batch, eigenvals, 1]
            filtered = spectral_signal * wavelet_filter
            spatial_filtered = torch.matmul(eigenvectors, filtered)
            filtered_signals.append(spatial_filtered)
        
        wavelet_features = torch.stack(filtered_signals, dim=-1)  # [batch, nodes, features, wavelets]
        
        return wavelet_features

class TemporalSpectralEnergyTracker(nn.Module):
    """
    Track spectral energy evolution over time and detect anomalies via energy trajectory changes
    """
    
    def __init__(self, num_eigenvals: int = 50, memory_length: int = 10):
        super().__init__()
        
        self.num_eigenvals = num_eigenvals
        self.memory_length = memory_length
        self._initialized = False
        self.trajectory_encoder = None
        self.anomaly_detector = None
        self.register_buffer('time_index', torch.tensor(0))
    
    def compute_spectral_energy(self, eigenvalues: torch.Tensor, 
                               node_features: torch.Tensor,
                               eigenvectors: torch.Tensor) -> torch.Tensor:
        """Compute spectral energy distribution"""
        
        # transform features to spectral domain
        spectral_features = torch.matmul(eigenvectors.transpose(-2, -1), node_features)
        
        # compute energy per eigenvalue
        energy_per_eigenval = torch.sum(spectral_features ** 2, dim=-1)  # [batch, eigenvals]
        
        # Normalize by eigenvalue (inverse relationship for anomaly detection)
        eigenval_weights = 1.0 / (eigenvalues + 1e-8)
        weighted_energy = energy_per_eigenval * eigenval_weights
        
        # Normalize to probability distribution
        energy_distribution = F.softmax(weighted_energy, dim=-1)
        
        return energy_distribution
    
    def update_energy_history(self, energy_distribution: torch.Tensor):
        """Update temporal energy history"""

        avg_energy = energy_distribution.mean(dim=0)
        if not self._initialized:
            actual_eigenvals = avg_energy.shape[0]
            self.register_buffer('energy_history', torch.zeros(self.memory_length, actual_eigenvals))
            
            self.trajectory_encoder = nn.LSTM(
                input_size=actual_eigenvals,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            
            self.anomaly_detector = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            self._initialized = True

        idx = self.time_index % self.memory_length
        self.energy_history[idx] = avg_energy
        self.time_index += 1
    
    def detect_energy_trajectory_anomalies(self, current_energy: torch.Tensor) -> torch.Tensor:
        """
        Detect anomalies based on energy trajectory deviation
        """
        
        batch_size = current_energy.shape[0]
        
        # check history
        if not self._initialized or not hasattr(self, 'energy_history'):
            return torch.zeros(batch_size, 1)
        
        # get history
        if self.time_index < self.memory_length:
            if self.time_index == 0:
                return torch.zeros(batch_size, 1)
            recent_history = self.energy_history[:self.time_index]
        else:
            start_idx = self.time_index % self.memory_length
            recent_history = torch.cat([
                self.energy_history[start_idx:],
                self.energy_history[:start_idx]
            ], dim=0)
        
        if recent_history.shape[0] < 2 or self.trajectory_encoder is None: # (no history)
            return torch.zeros(batch_size, 1)
        
        # sequences for LSTM
        sequences = recent_history.unsqueeze(0).repeat(batch_size, 1, 1)  
        lstm_out, _ = self.trajectory_encoder(sequences)
        trajectory_encoding = lstm_out[:, -1, :]  # Last timestep encoding

        # detect anomalies
        anomaly_scores = self.anomaly_detector(trajectory_encoding)
        
        return anomaly_scores
    
    def forward(self, eigenvalues: torch.Tensor, node_features: torch.Tensor,
                eigenvectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full temporal spectral energy analysis
        """
        
        energy_distribution = self.compute_spectral_energy(eigenvalues, node_features, eigenvectors)
        trajectory_anomalies = self.detect_energy_trajectory_anomalies(energy_distribution)
        self.update_energy_history(energy_distribution)
        
        return energy_distribution, trajectory_anomalies

class EnergyGuidedMultiScale(nn.Module):
    """
    Hierarchical spectral analysis with energy-guided routing
    Adaptive frequency band selection based on energy patterns
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 domain_type: str = 'general'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        num_wavelets = 8  
        
        self.coarse_wavelets = DynamicWaveletModule(
            num_wavelets=num_wavelets, 
            max_frequency=1.0,
            domain_type=domain_type
        )
        
        self.fine_wavelets = DynamicWaveletModule(
            num_wavelets=num_wavelets,
            max_frequency=2.0, 
            domain_type=domain_type
        )
        
        self.ultra_fine_wavelets = DynamicWaveletModule(
            num_wavelets=num_wavelets,
            max_frequency=4.0,
            domain_type=domain_type
        )
        
        # energy-based routing 
        self.energy_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 scales
            nn.Softmax(dim=-1)
        )
        
        # scale encoders
        self.coarse_encoder = nn.Linear(input_dim * num_wavelets, hidden_dim)   
        self.fine_encoder = nn.Linear(input_dim * num_wavelets, hidden_dim)     
        self.ultra_fine_encoder = nn.Linear(input_dim * num_wavelets, hidden_dim)
        
        # adaptive combination
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, graph_signal: torch.Tensor, eigenvalues: torch.Tensor,
                eigenvectors: torch.Tensor, energy_distribution: torch.Tensor) -> torch.Tensor:
        """
        Energy-guided multi-scale spectral analysis
        """
        
        batch_size, num_nodes, num_features = graph_signal.shape
        
        # multi-scale wavelets
        coarse_features = self.coarse_wavelets(graph_signal, eigenvalues, eigenvectors)
        fine_features = self.fine_wavelets(graph_signal, eigenvalues, eigenvectors) 
        ultra_fine_features = self.ultra_fine_wavelets(graph_signal, eigenvalues, eigenvectors)
        
        # flatten wavelet features
        coarse_flat = coarse_features.view(batch_size, num_nodes, -1)
        fine_flat = fine_features.view(batch_size, num_nodes, -1)
        ultra_fine_flat = ultra_fine_features.view(batch_size, num_nodes, -1)
        
        # encode each scale
        coarse_encoded = self.coarse_encoder(coarse_flat)
        fine_encoded = self.fine_encoder(fine_flat)
        ultra_fine_encoded = self.ultra_fine_encoder(ultra_fine_flat)
        
        # energy-guided routing weights - use mean energy distribution as routing signal
        routing_signal = energy_distribution.mean(dim=1)  # [batch, eigenvals] -> [batch, eigenvals_mean]
        if routing_signal.dim() > 1:
            routing_signal = routing_signal.mean(dim=-1, keepdim=True)  # [batch, 1]
        
        # expand to match input_dim for energy router
        if routing_signal.shape[-1] != self.input_dim:
            routing_signal = routing_signal.expand(-1, self.input_dim)    # [batch, input_dim]
        
        routing_weights = self.energy_router(routing_signal)  # [batch, 3]
        
        # apply routing weights
        routed_coarse = coarse_encoded * routing_weights[:, 0:1].unsqueeze(1)
        routed_fine = fine_encoded * routing_weights[:, 1:2].unsqueeze(1)
        routed_ultra_fine = ultra_fine_encoded * routing_weights[:, 2:3].unsqueeze(1)
        
        # combine scales
        combined_features = torch.cat([routed_coarse, routed_fine, routed_ultra_fine], dim=-1)
        output = self.combiner(combined_features)
        
        return output

class TemporalSpectralFusion(nn.Module):
    """
    Temporal-Spectral Fusion
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 temporal_steps: int = 10,
                 domain_type: str = 'general',
                 use_energy_tracking: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temporal_steps = temporal_steps
        self.domain_type = domain_type
        self.use_energy_tracking = use_energy_tracking

        self.multi_scale_spectral = EnergyGuidedMultiScale(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            domain_type=domain_type
        )
        
        if use_energy_tracking:
            self.energy_tracker = TemporalSpectralEnergyTracker(
                num_eigenvals=min(50, input_dim),
                memory_length=temporal_steps
            )
        
        # temporal modeling for spectral features
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # cross-time attention 
        self.cross_time_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional LSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # final fusion
        self.temporal_spectral_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # anomaly prediction
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Process temporal sequence of graphs with spectral analysis
        
        Args:
            temporal_graph_sequence: List of (node_features, eigenvalues, eigenvectors) for each timestep
        
        Returns:
            Dictionary with anomaly scores, energy distributions, and intermediate features
        """
        
        temporal_features = []
        energy_distributions = []
        trajectory_anomalies = []
        
        # Process each timestep
        for t, (node_features, eigenvalues, eigenvectors) in enumerate(temporal_graph_sequence):
 
            if self.use_energy_tracking:
                energy_dist, traj_anomaly = self.energy_tracker(eigenvalues, node_features, eigenvectors)
                energy_distributions.append(energy_dist)
                trajectory_anomalies.append(traj_anomaly)
            else:
                # compute energy distribution
                spectral_features = torch.matmul(eigenvectors.transpose(-2, -1), node_features)
                energy_dist = F.softmax(torch.sum(spectral_features ** 2, dim=-1), dim=-1)
                energy_distributions.append(energy_dist)
                trajectory_anomalies.append(torch.zeros(node_features.shape[0], 1))

            spectral_features = self.multi_scale_spectral(
                node_features, eigenvalues, eigenvectors, energy_dist
            )
            
            temporal_features.append(spectral_features)
        
        # stack temporal features
        temporal_sequence = torch.stack(temporal_features, dim=1)  # [batch, time, nodes, features]
        batch_size, time_steps, num_nodes, feature_dim = temporal_sequence.shape
        
        # reshape for temporal modeling
        temporal_reshaped = temporal_sequence.view(batch_size * num_nodes, time_steps, feature_dim)
        
        # temporal encoding
        temporal_encoded, _ = self.temporal_encoder(temporal_reshaped)
        
        # cross-time attention
        attended_features, attention_weights = self.cross_time_attention(
            temporal_encoded, temporal_encoded, temporal_encoded
        )

        fused_features = self.temporal_spectral_fusion(attended_features)
        final_features = fused_features[:, -1, :]  # [batch*nodes, features]
        
        # anomaly prediction
        anomaly_scores = self.anomaly_predictor(final_features)
        
        # reshape back to [batch, nodes]
        anomaly_scores = anomaly_scores.view(batch_size, num_nodes)
        
        # combine anomalies (if possible)
        if trajectory_anomalies:
            final_trajectory_anomaly = trajectory_anomalies[-1]  # Most recent
            combined_anomaly_scores = 0.7 * anomaly_scores + 0.3 * final_trajectory_anomaly.squeeze(-1)
        else:
            combined_anomaly_scores = anomaly_scores
        
        return {
            'anomaly_scores': combined_anomaly_scores,
            'spectral_features': final_features.view(batch_size, num_nodes, -1),
            'energy_distributions': torch.stack(energy_distributions, dim=1) if energy_distributions else None,
            'trajectory_anomalies': torch.stack(trajectory_anomalies, dim=1) if trajectory_anomalies else None,
            'attention_weights': attention_weights.view(batch_size, num_nodes, time_steps, time_steps)
        }

def create_temporal_spectral_model(config: Dict) -> TemporalSpectralFusion:
    """
    Factory function to create temporal-spectral fusion model
    """
    
    model = TemporalSpectralFusion(
        input_dim=config.get('input_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        temporal_steps=config.get('temporal_steps', 10),
        domain_type=config.get('domain_type', 'general'),
        use_energy_tracking=config.get('use_energy_tracking', True)
    )
    
    return model

if __name__ == "__main__":
    # example usage and testing
    print("Temporal-Spectral Fusion Architecture")
    print("=" * 50)
    
    config = {
        'input_dim': 64,
        'hidden_dim': 128, 
        'temporal_steps': 5,
        'domain_type': 'financial',
        'use_energy_tracking': True
    }
    
    model = create_temporal_spectral_model(config)
    
    # example temporal graph sequence
    batch_size, num_nodes, num_features = 32, 1000, 64
    temporal_sequence = []
    
    for t in range(config['temporal_steps']):
        # simulate graph data for timestep t
        node_features = torch.randn(batch_size, num_nodes, num_features)
        eigenvalues = torch.sort(torch.rand(batch_size, min(50, num_features)))[0]
        eigenvectors = torch.randn(batch_size, num_nodes, min(50, num_features))
        
        temporal_sequence.append((node_features, eigenvalues, eigenvectors))
    
    # Forward pass
    print(f"Processing temporal sequence of {len(temporal_sequence)} timesteps")
    
    with torch.no_grad():
        results = model(temporal_sequence)
    
    print(f"Results:")
    print(f"  Anomaly scores shape: {results['anomaly_scores'].shape}")
    print(f"  Spectral features shape: {results['spectral_features'].shape}")
    if results['energy_distributions'] is not None:
        print(f"  Energy distributions shape: {results['energy_distributions'].shape}")
    print(f"  Attention weights shape: {results['attention_weights'].shape}")
    