#!/usr/bin/env python3
"""
Model Wrapper for GraphSVX Integration
Provides unified interface for different GAD models including novel spectral methods

USAGE EXAMPLES:

# Distance-based model
wrapper = create_model_wrapper(
    model_type='distance_based',
    method_name='l2',
    train_data=X_train,
    train_labels=y_train
)

# Novel spectral model
from models.temporal_spectral_fusion import TemporalSpectralFusion
spectral_model = TemporalSpectralFusion(input_dim=64)
wrapper = create_model_wrapper(
    model=spectral_model,
    model_type='temporal_spectral'
)

# Or use convenience function
wrapper = create_spectral_model_wrapper(
    spectral_model=spectral_model,
    spectral_type='temporal'
)

# Use with GraphSVX
from utils.graphsvx import GraphSVXExplainer
explainer = GraphSVXExplainer(model=wrapper, data=data)
explanation = explainer.explain_node_comprehensive(node_idx=123)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - ModelWrapper - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class GADModelWrapper:
    """
    Wrapper class to provide unified interface for different GAD models
    Makes models compatible with GraphSVX explainer
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str = 'distance_based',
        device: str = 'cpu'
    ):
        """
        Initialize model wrapper
        
        Args:
            model: The GAD model to wrap
            model_type: Type of model ('distance_based', 'neural', 'ensemble')
            device: Device for computation
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.logger = setup_logging()
        
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.eval()
        
        self.logger.info(f"Initialized GAD model wrapper for {model_type} model")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            
        Returns:
            Model predictions
        """
        with torch.no_grad():
            if self.model_type == 'distance_based':
                return self._forward_distance_based(x, edge_index)
            elif self.model_type == 'neural':
                return self._forward_neural(x, edge_index)
            elif self.model_type == 'ensemble':
                return self._forward_ensemble(x, edge_index)
            elif self.model_type == 'contrastive_spectral':
                return self._forward_contrastive_spectral(x, edge_index)
            elif self.model_type == 'temporal_spectral':
                return self._forward_temporal_spectral(x, edge_index)
            elif self.model_type == 'explainable_spectral':
                return self._forward_explainable_spectral(x, edge_index)
            elif self.model_type == 'integrated_novel':
                return self._forward_integrated_novel(x, edge_index)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _forward_distance_based(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for distance-based models (L2, Mahalanobis, etc.)"""
        
        # for distance-based models, we need to compute anomaly scores
        # this is a simplified implementation - in practice, you'd use your actual distance methods
        
        if hasattr(self.model, 'compute_anomaly_scores'):
            scores = self.model.compute_anomaly_scores(x.cpu().numpy())
            return torch.tensor(scores, device=self.device, dtype=torch.float32)
        else:
            # Fallback: compute simple L2 distance from mean
            mean_features = x.mean(dim=0)
            distances = torch.norm(x - mean_features, dim=1)
            
            # Normalize to [0, 1] range
            min_dist, max_dist = distances.min(), distances.max()
            if max_dist > min_dist:
                scores = (distances - min_dist) / (max_dist - min_dist)
            else:
                scores = torch.zeros_like(distances)
            
            return scores
    
    def _forward_neural(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for neural network models (GAT, DSGAD, etc.)"""
        
        # standard neural network forward pass
        if hasattr(self.model, 'forward'):
            output = self.model(x, edge_index)
        else:
            output = self.model(x, edge_index)
        
        # handle different output formats
        if len(output.shape) == 2:
            if output.shape[1] == 1:
                # single output per node
                scores = torch.sigmoid(output.squeeze())
            else:
                # multi-class output, take probability of anomaly class
                scores = F.softmax(output, dim=1)[:, 1]
        else:
            # already single score per node
            scores = torch.sigmoid(output)
        
        return scores
    
    def _forward_ensemble(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for ensemble models"""
        
        if hasattr(self.model, 'predict_proba'):
            # scikit-learn style ensemble
            proba = self.model.predict_proba(x.cpu().numpy())
            if proba.shape[1] == 2:
                scores = torch.tensor(proba[:, 1], device=self.device, dtype=torch.float32)
            else:
                scores = torch.tensor(proba.squeeze(), device=self.device, dtype=torch.float32)
        else:
            # custom ensemble - assume it returns scores directly
            scores = self.model(x, edge_index)
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
        
        return scores
    
    def _forward_contrastive_spectral(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for contrastive spectral learning models"""
        
        try:
            # try different forward signatures for contrastive spectral models
            if hasattr(self.model, 'predict_anomaly_scores'):
                scores = self.model.predict_anomaly_scores(x, edge_index)
            elif hasattr(self.model, 'compute_anomaly_scores'):
                scores = self.model.compute_anomaly_scores(x, edge_index)
            elif hasattr(self.model, 'forward'):
                # standard forward pass
                output = self.model.forward(x, edge_index)
                # handle different output formats
                if isinstance(output, dict):
                    scores = output.get('anomaly_scores', output.get('scores', output.get('logits')))
                elif len(output.shape) == 2 and output.shape[1] == 1:
                    scores = torch.sigmoid(output.squeeze())
                else:
                    scores = torch.sigmoid(output)
            else:
                # fallback to model call
                scores = self.model(x, edge_index)
            
            # ensure tensor format
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
            
            # normalize to [0, 1] if needed
            if scores.min() < 0 or scores.max() > 1:
                scores = torch.sigmoid(scores)
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Contrastive spectral forward failed: {e}, using fallback")
            return self._fallback_spectral_forward(x, edge_index)
    
    def _forward_temporal_spectral(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for temporal spectral fusion models"""
        
        try:
            # temporal spectral models may need additional parameters
            if hasattr(self.model, 'predict_anomalies'):
                scores = self.model.predict_anomalies(x, edge_index)
            elif hasattr(self.model, 'forward') and hasattr(self.model, 'energy_tracker'):
                # models with energy tracking
                output = self.model.forward(x, edge_index)
                if isinstance(output, dict):
                    scores = output.get('anomaly_scores', output.get('energy_scores'))
                else:
                    scores = output
            else:
                # standard forward pass
                output = self.model(x, edge_index)
                scores = output if isinstance(output, torch.Tensor) else torch.tensor(output)
            
            # ensure proper format
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
            
            # handle multi-dimensional outputs (temporal models might return sequences)
            if len(scores.shape) > 1:
                if scores.shape[-1] == 1:
                    scores = scores.squeeze(-1)
                elif scores.shape[-1] > 1:
                    # take the last temporal step or mean across time
                    scores = scores.mean(dim=-1) if scores.shape[1] > 1 else scores[:, -1]
            
            return torch.sigmoid(scores)
            
        except Exception as e:
            self.logger.warning(f"Temporal spectral forward failed: {e}, using fallback")
            return self._fallback_spectral_forward(x, edge_index)
    
    def _forward_explainable_spectral(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for explainable spectral analysis models"""
        
        try:
            # explainable models may return explanations alongside predictions
            if hasattr(self.model, 'predict_with_explanations'):
                predictions, _ = self.model.predict_with_explanations(x, edge_index)
                scores = predictions
            elif hasattr(self.model, 'predict_anomaly_scores'):
                scores = self.model.predict_anomaly_scores(x, edge_index)
            else:
                output = self.model(x, edge_index)
                if isinstance(output, tuple):
                    # (predictions, explanations) tuple
                    scores, _ = output
                elif isinstance(output, dict):
                    scores = output.get('predictions', output.get('scores'))
                else:
                    scores = output
            
            # ensure tensor format
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
            
            return torch.sigmoid(scores)
            
        except Exception as e:
            self.logger.warning(f"Explainable spectral forward failed: {e}, using fallback")
            return self._fallback_spectral_forward(x, edge_index)
    
    def _forward_integrated_novel(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for integrated novel pipeline models"""
        
        try:
            # integrated models combine multiple approaches
            if hasattr(self.model, 'predict_integrated'):
                scores = self.model.predict_integrated(x, edge_index)
            elif hasattr(self.model, 'forward_complete'):
                output = self.model.forward_complete(x, edge_index)
                scores = output.get('final_scores') if isinstance(output, dict) else output
            else:
                # standard forward pass
                output = self.model(x, edge_index)
                if isinstance(output, dict):
                    # look for different possible score keys
                    scores = output.get('final_scores', 
                                     output.get('integrated_scores',
                                               output.get('anomaly_scores',
                                                         output.get('scores'))))
                else:
                    scores = output
            
            # handle ensemble-like outputs (integrated models may return multiple scores)
            if isinstance(scores, (list, tuple)):
                scores = torch.stack(scores).mean(dim=0)  # average ensemble scores
            
            # ensure tensor format
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
            
            return torch.sigmoid(scores)
            
        except Exception as e:
            self.logger.warning(f"Integrated novel forward failed: {e}, using fallback")
            return self._fallback_spectral_forward(x, edge_index)
    
    def _fallback_spectral_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Fallback method for spectral models when specific forwards fail"""
        
        try:
            # try basic forward call
            output = self.model(x, edge_index)
            
            # handle various output formats
            if isinstance(output, dict):
                scores = list(output.values())[0]  # first value
            elif isinstance(output, (list, tuple)):
                scores = output[0]  # first element
            else:
                scores = output
            
            # ensure tensor format
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
            
            # reshape if needed
            if len(scores.shape) > 1:
                scores = scores.squeeze() if scores.shape[-1] == 1 else scores.mean(dim=-1)
            
            return torch.sigmoid(scores)
            
        except Exception as e:
            self.logger.error(f"All spectral forward methods failed: {e}")
            # ultimate fallback: return uniform scores
            return torch.ones(x.shape[0], device=self.device) * 0.5
    
    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable"""
        return self.forward(x, edge_index)
    
    def to(self, device):
        """Move wrapper to device (for compatibility)"""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set to evaluation mode (for compatibility)"""
        if self.model is not None:
            self.model.eval()
        return self

class DistanceBasedModelWrapper(GADModelWrapper):
    """
    Specialized wrapper for distance-based anomaly detection methods
    """
    
    def __init__(
        self,
        method_name: str,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        device: str = 'cpu'
    ):
        """
        Initialize distance-based model wrapper
        
        Args:
            method_name: Name of distance method ('l2', 'mahalanobis', 'knn', 'pca')
            train_data: Training data for fitting the method
            train_labels: Training labels
            device: Device for computation
        """
        self.method_name = method_name.lower()
        self.train_data = train_data
        self.train_labels = train_labels
        self.device = device
        self.logger = setup_logging()
        
        # fit the method
        self._fit_method()
        
        # initialize parent - we'll override the model attribute
        self.logger = setup_logging()
        self.device = device
        self.model_type = 'distance_based'
        
        self.logger.info(f"Initialized {method_name} distance-based wrapper")
    
    def _fit_method(self):
        """Fit the distance-based method"""
        normal_data = self.train_data[self.train_labels == 0]
        
        if self.method_name == 'l2':
            self.normal_centroid = normal_data.mean(axis=0)
            self.anomaly_centroid = self.train_data[self.train_labels == 1].mean(axis=0)
        
        elif self.method_name == 'mahalanobis':
            self.normal_mean = normal_data.mean(axis=0)
            self.normal_cov = np.cov(normal_data.T) + np.eye(normal_data.shape[1]) * 1e-6
            self.inv_cov = np.linalg.inv(self.normal_cov)
        
        elif self.method_name == 'knn':
            from sklearn.neighbors import NearestNeighbors
            self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
            self.knn.fit(normal_data)
        
        elif self.method_name == 'pca':
            from sklearn.decomposition import PCA
            n_components = min(20, normal_data.shape[1], normal_data.shape[0])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(normal_data)
        
        else:
            raise ValueError(f"Unknown distance method: {self.method_name}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores using the distance method"""
        
        x_np = x.cpu().numpy()
        
        if self.method_name == 'l2':
            normal_distances = np.linalg.norm(x_np - self.normal_centroid, axis=1)
            anomaly_distances = np.linalg.norm(x_np - self.anomaly_centroid, axis=1)
            scores = normal_distances - anomaly_distances
        
        elif self.method_name == 'mahalanobis':
            diff = x_np - self.normal_mean
            scores = np.sum(diff @ self.inv_cov * diff, axis=1)
        
        elif self.method_name == 'knn':
            distances, _ = self.knn.kneighbors(x_np)
            scores = distances.mean(axis=1)
        
        elif self.method_name == 'pca':
            x_transformed = self.pca.transform(x_np)
            x_reconstructed = self.pca.inverse_transform(x_transformed)
            scores = np.sum((x_np - x_reconstructed) ** 2, axis=1)
        
        # normalize scores to [0, 1]
        scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
        min_score, max_score = scores.min(), scores.max()
        
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        
        return scores
    
    def to(self, device):
        """Move wrapper to device (for compatibility)"""
        self.device = device
        return self
    
    def eval(self):
        """Set to evaluation mode (for compatibility)"""
        return self

def create_model_wrapper(
    model: Optional[torch.nn.Module] = None,
    model_type: str = 'distance_based',
    method_name: Optional[str] = None,
    train_data: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    device: str = 'cpu'
) -> GADModelWrapper:
    """
    Factory function to create appropriate model wrapper
    
    Args:
        model: Neural network model (if applicable)
        model_type: Type of model wrapper ('distance_based', 'neural', 'ensemble', 
                   'contrastive_spectral', 'temporal_spectral', 'explainable_spectral', 'integrated_novel')
        method_name: Name of distance method (for distance-based models)
        train_data: Training data (for distance-based models)
        train_labels: Training labels (for distance-based models)
        device: Device for computation
        
    Returns:
        Appropriate model wrapper instance
    """
    
    # validate model_type
    valid_types = [
        'distance_based', 'neural', 'ensemble', 
        'contrastive_spectral', 'temporal_spectral', 
        'explainable_spectral', 'integrated_novel'
    ]
    
    if model_type not in valid_types:
        raise ValueError(f"model_type must be one of {valid_types}, got {model_type}")
    
    # distance-based models (special case - no model required)
    if model_type == 'distance_based' and method_name is not None:
        if train_data is None or train_labels is None:
            raise ValueError("train_data and train_labels required for distance-based models")
        return DistanceBasedModelWrapper(method_name, train_data, train_labels, device)
    
    # all other models require a model instance
    elif model is not None:
        return GADModelWrapper(model, model_type, device)
    
    else:
        raise ValueError(
            f"For model_type '{model_type}': "
            f"Either provide 'model' parameter, or "
            f"(for distance_based) provide method_name + train_data + train_labels"
        )

def create_spectral_model_wrapper(
    spectral_model: torch.nn.Module,
    spectral_type: str,
    device: str = 'cpu',
    **kwargs
) -> GADModelWrapper:
    """
    Convenience function specifically for creating spectral model wrappers
    
    Args:
        spectral_model: Instance of your novel spectral model
        spectral_type: Type of spectral model ('contrastive', 'temporal', 'explainable', 'integrated')
        device: Device for computation
        **kwargs: Additional arguments for the wrapper
        
    Returns:
        GADModelWrapper configured for spectral models
    """
    
    # map short names to full model types
    type_mapping = {
        'contrastive': 'contrastive_spectral',
        'temporal': 'temporal_spectral', 
        'explainable': 'explainable_spectral',
        'integrated': 'integrated_novel'
    }
    
    model_type = type_mapping.get(spectral_type, spectral_type)
    
    if model_type not in type_mapping.values():
        raise ValueError(f"spectral_type must be one of {list(type_mapping.keys())}, got {spectral_type}")
    
    return GADModelWrapper(spectral_model, model_type, device)