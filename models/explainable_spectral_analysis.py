#!/usr/bin/env python3
"""
Explainable Spectral Analysis
GraphSVX explainability integrated with dynamic spectral components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from itertools import combinations
import math

class SpectralShapleyExplainer(nn.Module):
    """
    Shapley values for dynamic spectral components
    Explain predictions using spectral wavelets and energy distributions
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_wavelets: int = 8,
                 max_coalitions: int = 1000,
                 sampling_strategy: str = 'adaptive'):
        super().__init__()
        
        self.model = model
        self.num_wavelets = num_wavelets
        self.max_coalitions = max_coalitions
        self.sampling_strategy = sampling_strategy
        
        # explanation aggregation
        self.explanation_aggregator = nn.Sequential(
            nn.Linear(num_wavelets * 3, 64),  # 3 views: spatial, spectral, temporal
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def generate_coalitions(self, feature_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate coalition masks for Shapley value computation
        Adaptive sampling focusing on spectral importance
        """
        
        num_features = feature_mask.shape[-1]
        coalitions = []
        
        if self.sampling_strategy == 'adaptive':
            # adaptive sampling based on spectral importance
            importance_scores = self._compute_spectral_importance(feature_mask)
            
            # sample more coalitions around important features
            for _ in range(self.max_coalitions):
                coalition_size = torch.randint(1, num_features, (1,)).item()
                
                # Weighted sampling based on importance
                probs = F.softmax(importance_scores, dim=-1)
                selected_indices = torch.multinomial(probs, coalition_size, replacement=False)
                
                coalition_mask = torch.zeros_like(feature_mask)
                coalition_mask[..., selected_indices] = 1.0
                
                coalitions.append(coalition_mask)
        
        elif self.sampling_strategy == 'exhaustive':
            for size in range(1, min(num_features + 1, int(math.log2(self.max_coalitions)) + 1)):
                for coalition in combinations(range(num_features), size):
                    if len(coalitions) >= self.max_coalitions:
                        break
                    
                    coalition_mask = torch.zeros_like(feature_mask)
                    coalition_mask[..., list(coalition)] = 1.0
                    coalitions.append(coalition_mask)
                
                if len(coalitions) >= self.max_coalitions:
                    break
        
        else: 
            # random sampling
            for _ in range(self.max_coalitions):
                coalition_mask = torch.bernoulli(torch.ones_like(feature_mask) * 0.5)
                coalitions.append(coalition_mask)
        
        return coalitions
    
    def _compute_spectral_importance(self, feature_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute initial spectral importance 
        """
        num_features = feature_mask.shape[-1]
        importance = torch.exp(-0.1 * torch.arange(num_features, dtype=torch.float))
        
        return importance
    
    def compute_marginal_contributions(self, 
                                     node_features: torch.Tensor,
                                     eigenvalues: torch.Tensor,
                                     eigenvectors: torch.Tensor,
                                     coalitions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute marginal contributions for each coalition
        """
        
        batch_size = node_features.shape[0]
        num_coalitions = len(coalitions)
        contributions = torch.zeros(batch_size, num_coalitions)
        
        # empty coalition
        baseline_input = torch.zeros_like(node_features)
        baseline_pred = self._predict_with_mask(baseline_input, eigenvalues, eigenvectors, torch.zeros_like(coalitions[0]))
        full_mask = torch.ones_like(coalitions[0])
        full_pred = self._predict_with_mask(node_features, eigenvalues, eigenvectors, full_mask)
        
        for i, coalition_mask in enumerate(coalitions):
            masked_prediction = self._predict_with_mask(node_features, eigenvalues, eigenvectors, coalition_mask)
            contributions[:, i] = masked_prediction.squeeze() - baseline_pred.squeeze()
        
        return contributions
    
    def _predict_with_mask(self, 
                          node_features: torch.Tensor,
                          eigenvalues: torch.Tensor, 
                          eigenvectors: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Make prediction with masked features
        """
        # apply mask to spectral features
        masked_features = node_features * mask.unsqueeze(1)  # Broadcast mask
        temporal_sequence = [(masked_features, eigenvalues, eigenvectors)]
        
        with torch.no_grad():
            results = self.model(temporal_sequence)
            prediction = results['anomaly_scores'].mean(dim=-1)  # Average over nodes
        
        return prediction
    
    def compute_shapley_values(self,
                              node_features: torch.Tensor,
                              eigenvalues: torch.Tensor,
                              eigenvectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Shapley values for spectral explanation
        """
        
        batch_size, num_nodes, num_features = node_features.shape
        feature_mask = torch.ones(num_features)
        coalitions = self.generate_coalitions(feature_mask)
        contributions = self.compute_marginal_contributions(
            node_features, eigenvalues, eigenvectors, coalitions
        )
        shapley_values = torch.zeros(batch_size, num_features)
        
        for i in range(num_features):
            containing_coalitions = []
            not_containing_coalitions = []
            
            for j, coalition in enumerate(coalitions):
                if coalition[i] == 1:
                    containing_coalitions.append(j)
                else:
                    not_containing_coalitions.append(j)
            
            if containing_coalitions and not_containing_coalitions:
                with_i = contributions[:, containing_coalitions].mean(dim=1)
                without_i = contributions[:, not_containing_coalitions].mean(dim=1)
                shapley_values[:, i] = with_i - without_i
        
        return {
            'shapley_values': shapley_values,
            'coalition_contributions': contributions,
            'feature_importance_ranking': torch.argsort(shapley_values.abs().mean(dim=0), descending=True)
        }

class MultiViewSpectralExplainer(nn.Module):
    """
    Multi-view explainability for spectral, spatial, and temporal components
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_wavelets: int = 8,
                 consistency_threshold: float = 0.7):
        super().__init__()
        
        self.model = model
        self.num_wavelets = num_wavelets
        self.consistency_threshold = consistency_threshold
        self.spectral_explainer = SpectralShapleyExplainer(model, num_wavelets)
        self.spatial_explainer = SpectralShapleyExplainer(model, num_wavelets)
        self.temporal_explainer = SpectralShapleyExplainer(model, num_wavelets)
        self.consistency_validator = nn.Sequential(
            nn.Linear(num_wavelets * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def explain_prediction(self,
                          temporal_graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                          target_node_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate multi-view explanations for predictions
        """
        
        with torch.no_grad():
            results = self.model(temporal_graph_sequence)
            predictions = results['anomaly_scores']
            attention_weights = results.get('attention_weights', None)
        
        if target_node_idx is not None:
            node_predictions = predictions[:, target_node_idx]
        else:
            node_predictions = predictions.mean(dim=1)  # Average over nodes
        
        explanations = {}

        latest_features, latest_eigenvals, latest_eigenvecs = temporal_graph_sequence[-1]
        spectral_explanation = self.spectral_explainer.compute_shapley_values(
            latest_features, latest_eigenvals, latest_eigenvecs
        )
        explanations['spectral'] = spectral_explanation

        spatial_explanation = self.spatial_explainer.compute_shapley_values(
            latest_features, latest_eigenvals, latest_eigenvecs
        )
        explanations['spatial'] = spatial_explanation
        
        # Temporal view explanation
        if len(temporal_graph_sequence) > 1:
            # Use temporal differences
            temporal_diff = latest_features - temporal_graph_sequence[0][0]
            temporal_explanation = self.temporal_explainer.compute_shapley_values(
                temporal_diff, latest_eigenvals, latest_eigenvecs
            )
            explanations['temporal'] = temporal_explanation
        else:
            explanations['temporal'] = spatial_explanation  # Fallback
        
        consistency_scores = self.validate_cross_view_consistency(explanations)
        explanations['consistency'] = consistency_scores
        integrated_explanation = self.integrate_multi_view_explanations(explanations)
        explanations['integrated'] = integrated_explanation
        
        if attention_weights is not None:
            explanations['attention_temporal'] = self.explain_temporal_attention(
                attention_weights, temporal_graph_sequence
            )
        
        return explanations
    
    def validate_cross_view_consistency(self, explanations: Dict) -> Dict[str, torch.Tensor]:
        """
        Validate consistency across different explanation views
        """
        
        spectral_shapley = explanations['spectral']['shapley_values']
        spatial_shapley = explanations['spatial']['shapley_values']
        temporal_shapley = explanations['temporal']['shapley_values']
        
        # pairwise correlations
        spectral_spatial_corr = self._compute_explanation_correlation(spectral_shapley, spatial_shapley)
        spectral_temporal_corr = self._compute_explanation_correlation(spectral_shapley, temporal_shapley)
        spatial_temporal_corr = self._compute_explanation_correlation(spatial_shapley, temporal_shapley)
        

        consistency_input = torch.cat([
            spectral_shapley, spatial_shapley, temporal_shapley
        ], dim=-1)
        
        overall_consistency = self.consistency_validator(consistency_input)
        
        return {
            'spectral_spatial_correlation': spectral_spatial_corr,
            'spectral_temporal_correlation': spectral_temporal_corr, 
            'spatial_temporal_correlation': spatial_temporal_corr,
            'overall_consistency': overall_consistency,
            'is_consistent': overall_consistency > self.consistency_threshold
        }
    
    def _compute_explanation_correlation(self, explanation1: torch.Tensor, explanation2: torch.Tensor) -> torch.Tensor:
        """Compute correlation between two explanation vectors"""
        
        # normalize
        exp1_norm = F.normalize(explanation1, p=2, dim=-1)
        exp2_norm = F.normalize(explanation2, p=2, dim=-1)
        
        # cosine similarity
        correlation = torch.sum(exp1_norm * exp2_norm, dim=-1)
        
        return correlation
    
    def integrate_multi_view_explanations(self, explanations: Dict) -> Dict[str, torch.Tensor]:
        """
        Integrate explanations from multiple views
        """
        
        spectral_shapley = explanations['spectral']['shapley_values']
        spatial_shapley = explanations['spatial']['shapley_values']
        temporal_shapley = explanations['temporal']['shapley_values']
        
        consistency_scores = explanations['consistency']['overall_consistency']
        
        consistency_weights = F.softmax(consistency_scores.unsqueeze(-1), dim=0)
        
        integrated_shapley = (
            consistency_weights * spectral_shapley +
            consistency_weights * spatial_shapley + 
            consistency_weights * temporal_shapley
        ) / 3.0
        
        integrated_importance = torch.argsort(integrated_shapley.abs().mean(dim=0), descending=True)
        
        return {
            'integrated_shapley_values': integrated_shapley,
            'integrated_feature_importance': integrated_importance,
            'view_weights': consistency_weights.squeeze()
        }
    
    def explain_temporal_attention(self, 
                                  attention_weights: torch.Tensor,
                                  temporal_graph_sequence: List) -> Dict[str, torch.Tensor]:
        """
        Explain temporal attention patterns
        """
        
        batch_size, num_nodes, num_timesteps, _ = attention_weights.shape
        temporal_importance = attention_weights.mean(dim=[1, 3])  
        node_temporal_patterns = attention_weights.mean(dim=3)  
        critical_timesteps = torch.argmax(temporal_importance, dim=-1)
        
        return {
            'temporal_importance': temporal_importance,
            'node_temporal_patterns': node_temporal_patterns,
            'critical_timesteps': critical_timesteps,
            'attention_entropy': self._compute_attention_entropy(attention_weights)
        }
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions"""
        
        normalized_attention = F.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(normalized_attention * torch.log(normalized_attention + 1e-8), dim=-1)
        return entropy.mean(dim=[1, 2])  
    
class SpectralExplanationVisualizer:
    """
    Visualization utilities for spectral explanations
    """
    
    @staticmethod
    def create_explanation_summary(explanations: Dict) -> Dict[str, str]:
        """
        Create readable explanation summary
        """
        integrated = explanations.get('integrated', {})
        consistency = explanations.get('consistency', {})
        shapley_values = integrated.get('integrated_shapley_values', None)
        feature_importance = integrated.get('integrated_feature_importance', None)
        summary = {
            'explanation_type': 'Multi-view Spectral Explanation',
            'consistency_status': 'High' if consistency.get('overall_consistency', torch.tensor(0.0)).mean() > 0.7 else 'Low',
            'top_features': f"Features {feature_importance[:5].tolist()}" if feature_importance is not None else "N/A",
            'explanation_confidence': f"{consistency.get('overall_consistency', torch.tensor(0.0)).mean():.3f}",
            'dominant_view': 'Spectral' if explanations.get('spectral') else 'Spatial'
        }
        
        return summary

def create_explainable_spectral_model(model: nn.Module, config: Dict) -> MultiViewSpectralExplainer:
    """
    Factory function to create explainable spectral model
    """
    
    explainer = MultiViewSpectralExplainer(
        model=model,
        num_wavelets=config.get('num_wavelets', 8),
        consistency_threshold=config.get('consistency_threshold', 0.7)
    )
    
    return explainer

if __name__ == "__main__":
    print("Explainable Spectral Analysis")
    print("=" * 40)
    from temporal_spectral_fusion import create_temporal_spectral_model
    model_config = {
        'input_dim': 64,
        'hidden_dim': 128,
        'temporal_steps': 5,
        'domain_type': 'financial'
    }

    base_model = create_temporal_spectral_model(model_config)
    explainer_config = {
        'num_wavelets': 8,
        'consistency_threshold': 0.7
    }
    explainer = create_explainable_spectral_model(base_model, explainer_config)
    batch_size, num_nodes, num_features = 4, 100, 64
    temporal_sequence = []
    
    for t in range(3):
        node_features = torch.randn(batch_size, num_nodes, num_features)
        eigenvalues = torch.sort(torch.rand(batch_size, 20))[0]
        eigenvectors = torch.randn(batch_size, num_nodes, 20)
        temporal_sequence.append((node_features, eigenvalues, eigenvectors))
    
    with torch.no_grad():
        explanations = explainer.explain_prediction(temporal_sequence, target_node_idx=0)

    summary = SpectralExplanationVisualizer.create_explanation_summary(explanations)
    print("Explanation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    