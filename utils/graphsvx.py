#!/usr/bin/env python3
"""
GraphSVX: Graph Shapley Value Explainer for GAD Pipeline
Provides interpretable explanations for anomaly detection predictions using Shapley values
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from itertools import combinations, chain
import logging
from tqdm import tqdm
import copy
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - GraphSVX - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class GraphSVXExplainer:
    """
    GraphSVX: Graph Shapley Value Explainer
    
    Computes Shapley values to explain GNN predictions for anomaly detection.
    Provides fair attribution of features and graph structure to predictions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        num_samples: int = 1000,
        max_coalition_size: Optional[int] = None,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize GraphSVX explainer
        
        Args:
            model: Trained GNN model to explain
            data: Graph data (PyTorch Geometric Data object)
            num_samples: Number of coalition samples for Shapley approximation
            max_coalition_size: Maximum size of coalitions to consider
            device: Device for computation ('cpu' or 'cuda')
            verbose: Whether to show progress bars
        """
        self.model = model
        self.data = data
        self.num_samples = num_samples
        self.max_coalition_size = max_coalition_size
        self.device = device
        self.verbose = verbose
        self.logger = setup_logging()
        
        # move model and data to device
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        
        # set model to evaluation mode
        self.model.eval()
        
        # cache original prediction for efficiency
        self._original_predictions = None
        
        self.logger.info(f"GraphSVX initialized for {data.num_nodes} nodes, {data.num_features} features")
    
    def _get_model_prediction(self, perturbed_data: Data, node_idx: int) -> float:
        """
        Get model prediction for a specific node with perturbed data
        
        Args:
            perturbed_data: Perturbed graph data
            node_idx: Index of node to explain
            
        Returns:
            Model prediction score for the node
        """
        with torch.no_grad():
            # get model output
            if hasattr(self.model, 'forward'):
                output = self.model(perturbed_data.x, perturbed_data.edge_index)
            else:
                # handle different model architectures
                output = self.model(perturbed_data)
            
            # handle different output formats
            if len(output.shape) == 2:  # [num_nodes, num_classes]
                if output.shape[1] == 1:  # Binary classification with single output
                    score = torch.sigmoid(output[node_idx, 0]).item()
                else:  # Multi-class or binary with 2 outputs
                    score = F.softmax(output[node_idx], dim=0)[1].item()  # Probability of anomaly class
            else:  # [num_nodes] - single score per node
                score = torch.sigmoid(output[node_idx]).item()
            
            return score
    
    def _create_feature_perturbation(self, coalition: List[int], baseline_value: float = 0.0) -> Data:
        """
        Create perturbed data by masking features not in coalition
        
        Args:
            coalition: List of feature indices to keep
            baseline_value: Value to use for masked features
            
        Returns:
            Perturbed data object
        """
        perturbed_data = copy.deepcopy(self.data)
        
        # create feature mask
        feature_mask = torch.zeros(self.data.num_features, dtype=torch.bool, device=self.device)
        if coalition:
            feature_mask[coalition] = True
        
        # apply perturbation - mask out features not in coalition
        perturbed_data.x = perturbed_data.x.clone()
        perturbed_data.x[:, ~feature_mask] = baseline_value
        
        return perturbed_data
    
    def _create_edge_perturbation(self, node_idx: int, coalition: List[int]) -> Data:
        """
        Create perturbed data by keeping only edges to nodes in coalition
        
        Args:
            node_idx: Central node index
            coalition: List of neighbor node indices to keep edges to
            
        Returns:
            Perturbed data object
        """
        perturbed_data = copy.deepcopy(self.data)
        
        if not coalition:
            # remove all edges involving the node
            edge_mask = (perturbed_data.edge_index[0] != node_idx) & (perturbed_data.edge_index[1] != node_idx)
            perturbed_data.edge_index = perturbed_data.edge_index[:, edge_mask]
            return perturbed_data
        
        # find edges involving the central node
        node_edges_mask = (perturbed_data.edge_index[0] == node_idx) | (perturbed_data.edge_index[1] == node_idx)
        node_edges = perturbed_data.edge_index[:, node_edges_mask]
        
        # keep only edges to nodes in coalition
        coalition_set = set(coalition)
        keep_edges_mask = torch.zeros(node_edges.shape[1], dtype=torch.bool, device=self.device)
        
        for i in range(node_edges.shape[1]):
            edge = node_edges[:, i]
            other_node = edge[1].item() if edge[0].item() == node_idx else edge[0].item()
            if other_node in coalition_set:
                keep_edges_mask[i] = True
        
        # reconstruct edge index
        kept_node_edges = node_edges[:, keep_edges_mask]
        other_edges = perturbed_data.edge_index[:, ~node_edges_mask]
        perturbed_data.edge_index = torch.cat([other_edges, kept_node_edges], dim=1)
        
        return perturbed_data
    
    def _generate_coalitions(self, feature_indices: List[int], sample_size: Optional[int] = None) -> List[List[int]]:
        """
        Generate coalition samples for Shapley value computation
        
        Args:
            feature_indices: List of all feature/node indices to consider
            sample_size: Number of coalitions to sample (if None, use self.num_samples)
            
        Returns:
            List of coalitions (each coalition is a list of indices)
        """
        if sample_size is None:
            sample_size = self.num_samples
            
        coalitions = []
        n_features = len(feature_indices)
        
        # always include empty coalition and full coalition
        coalitions.append([])
        coalitions.append(feature_indices.copy())
        
        # generate random coalitions of different sizes
        for _ in range(sample_size - 2):
            coalition_size = np.random.randint(0, n_features + 1)
            if coalition_size == 0:
                coalition = []
            elif coalition_size == n_features:
                coalition = feature_indices.copy()
            else:
                coalition = np.random.choice(feature_indices, size=coalition_size, replace=False).tolist()
            coalitions.append(coalition)
        
        return coalitions
    
    def _compute_shapley_values(
        self,
        node_idx: int,
        coalitions: List[List[int]],
        perturbation_type: str = 'feature'
    ) -> Dict[int, float]:
        """
        Compute Shapley values using coalition-based sampling
        
        Args:
            node_idx: Index of node to explain
            coalitions: List of coalitions to evaluate
            perturbation_type: Type of perturbation ('feature' or 'edge')
            
        Returns:
            Dictionary mapping feature/node indices to Shapley values
        """
        # get all unique indices from coalitions
        all_indices = set()
        for coalition in coalitions:
            all_indices.update(coalition)
        all_indices = sorted(list(all_indices))
        
        # initialize Shapley values
        shapley_values = {idx: 0.0 for idx in all_indices}
        
        # evaluate all coalitions
        coalition_values = {}
        
        if self.verbose:
            coalition_iter = tqdm(coalitions, desc=f"Evaluating coalitions for node {node_idx}")
        else:
            coalition_iter = coalitions
        
        for coalition in coalition_iter:
            coalition_key = tuple(sorted(coalition))
            
            if coalition_key not in coalition_values:
                if perturbation_type == 'feature':
                    perturbed_data = self._create_feature_perturbation(coalition)
                else:  # edge perturbation
                    perturbed_data = self._create_edge_perturbation(node_idx, coalition)
                
                coalition_values[coalition_key] = self._get_model_prediction(perturbed_data, node_idx)
        
        # compute marginal contributions
        for idx in all_indices:
            contributions = []
            
            for coalition in coalitions:
                if idx in coalition:
                    # marginal contribution when adding idx to coalition without idx
                    coalition_without_idx = [i for i in coalition if i != idx]
                    
                    coalition_key = tuple(sorted(coalition))
                    coalition_without_key = tuple(sorted(coalition_without_idx))
                    
                    if coalition_key in coalition_values and coalition_without_key in coalition_values:
                        contribution = coalition_values[coalition_key] - coalition_values[coalition_without_key]
                        contributions.append(contribution)
            
            if contributions:
                shapley_values[idx] = np.mean(contributions)
        
        return shapley_values
    
    def explain_node_features(
        self,
        node_idx: int,
        baseline_value: float = 0.0
    ) -> Dict[str, Union[Dict[int, float], float]]:
        """
        Explain node prediction using feature-based Shapley values
        
        Args:
            node_idx: Index of node to explain
            baseline_value: Baseline value for feature perturbations
            
        Returns:
            Dictionary containing Shapley values and explanation metadata
        """
        self.logger.info(f"Explaining node {node_idx} features...")
        
        # get original prediction
        original_prediction = self._get_model_prediction(self.data, node_idx)
        
        # generate feature coalitions
        feature_indices = list(range(self.data.num_features))
        coalitions = self._generate_coalitions(feature_indices)
        
        # compute Shapley values
        shapley_values = self._compute_shapley_values(node_idx, coalitions, 'feature')
        
        # get baseline prediction (all features masked)
        baseline_data = self._create_feature_perturbation([], baseline_value)
        baseline_prediction = self._get_model_prediction(baseline_data, node_idx)
        
        explanation = {
            'node_idx': node_idx,
            'original_prediction': original_prediction,
            'baseline_prediction': baseline_prediction,
            'feature_shapley_values': shapley_values,
            'explanation_type': 'feature',
            'num_coalitions': len(coalitions),
            'baseline_value': baseline_value
        }
        
        return explanation
    
    def explain_node_structure(
        self,
        node_idx: int,
        k_hop: int = 2
    ) -> Dict[str, Union[Dict[int, float], float, List[int]]]:
        """
        Explain node prediction using structural (edge-based) Shapley values
        
        Args:
            node_idx: Index of node to explain
            k_hop: Number of hops to consider for neighborhood
            
        Returns:
            Dictionary containing Shapley values and explanation metadata
        """
        self.logger.info(f"Explaining node {node_idx} structure...")
        
        # get original prediction
        original_prediction = self._get_model_prediction(self.data, node_idx)
        
        # get k-hop neighborhood - ensure node_idx is tensor
        node_idx_tensor = torch.tensor(node_idx, device=self.device) if not isinstance(node_idx, torch.Tensor) else node_idx
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx_tensor, k_hop, self.data.edge_index, relabel_nodes=False
        )
        
        # get neighbor nodes (excluding the central node)
        neighbors = [n.item() for n in subset if n.item() != node_idx]
        
        if not neighbors:
            self.logger.warning(f"Node {node_idx} has no neighbors in {k_hop}-hop subgraph")
            return {
                'node_idx': node_idx,
                'original_prediction': original_prediction,
                'baseline_prediction': original_prediction,
                'structure_shapley_values': {},
                'explanation_type': 'structure',
                'neighbors': [],
                'k_hop': k_hop,
                'num_coalitions': 0
            }
        
        # generate neighbor coalitions
        coalitions = self._generate_coalitions(neighbors)
        
        # compute Shapley values
        shapley_values = self._compute_shapley_values(node_idx, coalitions, 'edge')
        
        # get baseline prediction (node isolated)
        baseline_data = self._create_edge_perturbation(node_idx, [])
        baseline_prediction = self._get_model_prediction(baseline_data, node_idx)
        
        explanation = {
            'node_idx': node_idx,
            'original_prediction': original_prediction,
            'baseline_prediction': baseline_prediction,
            'structure_shapley_values': shapley_values,
            'explanation_type': 'structure',
            'neighbors': neighbors,
            'k_hop': k_hop,
            'num_coalitions': len(coalitions)
        }
        
        return explanation
    
    def explain_node_comprehensive(
        self,
        node_idx: int,
        k_hop: int = 2,
        baseline_value: float = 0.0
    ) -> Dict[str, Union[Dict[int, float], float, List[int]]]:
        """
        Provide comprehensive explanation combining features and structure
        
        Args:
            node_idx: Index of node to explain
            k_hop: Number of hops for structural explanation
            baseline_value: Baseline value for feature perturbations
            
        Returns:
            Combined explanation dictionary
        """
        self.logger.info(f"Providing comprehensive explanation for node {node_idx}...")
        
        # get both explanations
        feature_explanation = self.explain_node_features(node_idx, baseline_value)
        structure_explanation = self.explain_node_structure(node_idx, k_hop)
        
        # combine explanations
        comprehensive_explanation = {
            'node_idx': node_idx,
            'original_prediction': feature_explanation['original_prediction'],
            'feature_explanation': feature_explanation,
            'structure_explanation': structure_explanation,
            'explanation_type': 'comprehensive'
        }
        
        return comprehensive_explanation
    
    def explain_batch(
        self,
        node_indices: List[int],
        explanation_type: str = 'comprehensive',
        **kwargs
    ) -> Dict[int, Dict]:
        """
        Explain multiple nodes in batch
        
        Args:
            node_indices: List of node indices to explain
            explanation_type: Type of explanation ('feature', 'structure', 'comprehensive')
            **kwargs: Additional arguments for explanation methods
            
        Returns:
            Dictionary mapping node indices to their explanations
        """
        explanations = {}
        
        if self.verbose:
            node_iter = tqdm(node_indices, desc="Explaining nodes")
        else:
            node_iter = node_indices
        
        for node_idx in node_iter:
            try:
                if explanation_type == 'feature':
                    explanation = self.explain_node_features(node_idx, **kwargs)
                elif explanation_type == 'structure':
                    explanation = self.explain_node_structure(node_idx, **kwargs)
                else:  # comprehensive
                    explanation = self.explain_node_comprehensive(node_idx, **kwargs)
                
                explanations[node_idx] = explanation
                
            except Exception as e:
                self.logger.error(f"Failed to explain node {node_idx}: {e}")
                explanations[node_idx] = {'error': str(e)}
        
        return explanations
    
    def get_top_features(
        self,
        explanation: Dict,
        top_k: int = 5,
        by_magnitude: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get top-k features from explanation
        
        Args:
            explanation: Explanation dictionary
            top_k: Number of top features to return
            by_magnitude: Whether to sort by absolute value
            
        Returns:
            List of (feature_index, shapley_value) tuples
        """
        if 'feature_shapley_values' not in explanation:
            return []
        
        shapley_values = explanation['feature_shapley_values']
        
        if by_magnitude:
            items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            items = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
        
        return items[:top_k]
    
    def get_top_neighbors(
        self,
        explanation: Dict,
        top_k: int = 5,
        by_magnitude: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get top-k neighbors from structural explanation
        
        Args:
            explanation: Explanation dictionary
            top_k: Number of top neighbors to return
            by_magnitude: Whether to sort by absolute value
            
        Returns:
            List of (neighbor_index, shapley_value) tuples
        """
        if 'structure_shapley_values' not in explanation:
            return []
        
        shapley_values = explanation['structure_shapley_values']
        
        if by_magnitude:
            items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            items = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
        
        return items[:top_k]