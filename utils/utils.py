#!/usr/bin/env python3
"""
Explanation Utilities for GraphSVX
Provides visualization and reporting capabilities for explanations
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import networkx as nx
from torch_geometric.utils import to_networkx
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - ExplainerUtils - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class ExplanationVisualizer:
    """
    Visualization utilities for graph explanations
    """
    
    def __init__(self, figsize=(12, 8), style='whitegrid'):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
            style: Seaborn style
        """
        self.figsize = figsize
        self.logger = setup_logging()
        
        # set style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_feature_importance(
        self,
        explanation: Dict,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance from Shapley values
        
        Args:
            explanation: Explanation dictionary
            feature_names: List of feature names (if None, use indices)
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if 'feature_shapley_values' not in explanation:
            raise ValueError("Explanation does not contain feature Shapley values")
        
        shapley_values = explanation['feature_shapley_values']
        
        # sort by absolute value
        sorted_items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_items = sorted_items[:top_k]
        
        # prepare data
        indices, values = zip(*top_items) if top_items else ([], [])
        
        if feature_names:
            labels = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
        else:
            labels = [f'Feature_{i}' for i in indices]
        
        # create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # color bars by positive/negative values
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7)
        
        # customize plot
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Shapley Value')
        
        if title is None:
            node_idx = explanation.get('node_idx', 'Unknown')
            title = f'Feature Importance for Node {node_idx}'
        ax.set_title(title)
        
        # add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                   ha='left' if value >= 0 else 'right', va='center')
        
        # add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Positive contribution'),
            Patch(facecolor='red', alpha=0.7, label='Negative contribution')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved feature importance plot to {save_path}")
        
        return fig
    
    def plot_neighbor_importance(
        self,
        explanation: Dict,
        node_names: Optional[Dict[int, str]] = None,
        top_k: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot neighbor importance from structural Shapley values
        
        Args:
            explanation: Explanation dictionary
            node_names: Dictionary mapping node indices to names
            top_k: Number of top neighbors to show
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if 'structure_shapley_values' not in explanation:
            raise ValueError("Explanation does not contain structure Shapley values")
        
        shapley_values = explanation['structure_shapley_values']
        
        if not shapley_values:
            self.logger.warning("No structural Shapley values to plot")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No neighbors to explain', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or 'Neighbor Importance')
            return fig
        
        # sort by absolute value
        sorted_items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_items = sorted_items[:top_k]
        
        # prepare data
        indices, values = zip(*top_items)
        
        if node_names:
            labels = [node_names.get(i, f'Node_{i}') for i in indices]
        else:
            labels = [f'Node_{i}' for i in indices]
        
        # create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # color bars by positive/negative values
        colors = ['red' if v < 0 else 'green' for v in values]
        
        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7)
        
        # customize plot
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Shapley Value')
        
        if title is None:
            node_idx = explanation.get('node_idx', 'Unknown')
            title = f'Neighbor Importance for Node {node_idx}'
        ax.set_title(title)
        
        # add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                   ha='left' if value >= 0 else 'right', va='center')
        
        # add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Positive influence'),
            Patch(facecolor='red', alpha=0.7, label='Negative influence')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved neighbor importance plot to {save_path}")
        
        return fig
    
    def plot_subgraph_explanation(
        self,
        data,
        explanation: Dict,
        k_hop: int = 2,
        node_size_factor: float = 300,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot subgraph with node importance highlighting
        
        Args:
            data: Graph data
            explanation: Explanation dictionary
            k_hop: Number of hops for subgraph
            node_size_factor: Factor for node size scaling
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if 'structure_shapley_values' not in explanation:
            raise ValueError("Explanation does not contain structure Shapley values")
        
        node_idx = explanation['node_idx']
        shapley_values = explanation['structure_shapley_values']
        
        # get k-hop subgraph
        from torch_geometric.utils import k_hop_subgraph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, k_hop, data.edge_index, relabel_nodes=True
        )
        
        # create networkx graph
        G = nx.Graph()
        
        # add nodes
        for i, original_idx in enumerate(subset):
            G.add_node(i, original_idx=original_idx.item())
        
        # add edges
        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)
        
        # create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # node colors and sizes based on Shapley values
        node_colors = []
        node_sizes = []
        
        central_node_mapped = mapping.item()  # Mapped index of central node
        
        for i in G.nodes():
            original_idx = G.nodes[i]['original_idx']
            
            if i == central_node_mapped:
                # central node - highlight in special color
                node_colors.append('gold')
                node_sizes.append(node_size_factor * 2)
            elif original_idx in shapley_values:
                # neighbor with Shapley value
                shap_val = shapley_values[original_idx]
                if shap_val > 0:
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightcoral')
                node_sizes.append(node_size_factor * (1 + abs(shap_val)))
            else:
                # other nodes
                node_colors.append('lightgray')
                node_sizes.append(node_size_factor)
        
        # draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        # add labels for important nodes
        labels = {}
        for i in G.nodes():
            original_idx = G.nodes[i]['original_idx']
            if i == central_node_mapped:
                labels[i] = f'{original_idx}*'
            elif original_idx in shapley_values and abs(shapley_values[original_idx]) > 0.01:
                labels[i] = f'{original_idx}'
        
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        # customize plot
        if title is None:
            title = f'Subgraph Explanation for Node {node_idx}'
        ax.set_title(title)
        ax.axis('off')
        
        # add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', label=f'Central Node ({node_idx})'),
            Patch(facecolor='lightgreen', label='Positive influence'),
            Patch(facecolor='lightcoral', label='Negative influence'),
            Patch(facecolor='lightgray', label='Other nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved subgraph explanation plot to {save_path}")
        
        return fig
    
    def plot_comprehensive_explanation(
        self,
        explanation: Dict,
        feature_names: Optional[List[str]] = None,
        node_names: Optional[Dict[int, str]] = None,
        top_k: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive explanation combining features and structure
        
        Args:
            explanation: Comprehensive explanation dictionary
            feature_names: List of feature names
            node_names: Dictionary mapping node indices to names
            top_k: Number of top items to show for each type
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if explanation.get('explanation_type') != 'comprehensive':
            raise ValueError("Explanation must be comprehensive type")
        
        feature_exp = explanation['feature_explanation']
        structure_exp = explanation['structure_explanation']
        
        # create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # plot feature importance
        if 'feature_shapley_values' in feature_exp:
            shapley_values = feature_exp['feature_shapley_values']
            sorted_items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
            top_items = sorted_items[:top_k]
            
            if top_items:
                indices, values = zip(*top_items)
                if feature_names:
                    labels = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
                else:
                    labels = [f'Feature_{i}' for i in indices]
                
                colors = ['red' if v < 0 else 'blue' for v in values]
                ax1.barh(range(len(labels)), values, color=colors, alpha=0.7)
                ax1.set_yticks(range(len(labels)))
                ax1.set_yticklabels(labels)
                ax1.set_xlabel('Shapley Value')
                ax1.set_title('Feature Importance')
        
        # plot neighbor importance
        if 'structure_shapley_values' in structure_exp:
            shapley_values = structure_exp['structure_shapley_values']
            if shapley_values:
                sorted_items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
                top_items = sorted_items[:top_k]
                
                if top_items:
                    indices, values = zip(*top_items)
                    if node_names:
                        labels = [node_names.get(i, f'Node_{i}') for i in indices]
                    else:
                        labels = [f'Node_{i}' for i in indices]
                    
                    colors = ['red' if v < 0 else 'green' for v in values]
                    ax2.barh(range(len(labels)), values, color=colors, alpha=0.7)
                    ax2.set_yticks(range(len(labels)))
                    ax2.set_yticklabels(labels)
                    ax2.set_xlabel('Shapley Value')
                    ax2.set_title('Neighbor Importance')
                else:
                    ax2.text(0.5, 0.5, 'No neighbors to explain', 
                           ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Neighbor Importance')
            else:
                ax2.text(0.5, 0.5, 'No structural explanation available', 
                       ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Neighbor Importance')
        
        node_idx = explanation.get('node_idx', 'Unknown')
        fig.suptitle(f'Comprehensive Explanation for Node {node_idx}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comprehensive explanation plot to {save_path}")
        
        return fig

class ExplanationReporter:
    """
    Generate structured reports from explanations
    """
    
    def __init__(self):
        self.logger = setup_logging()
    
    def generate_text_report(
        self,
        explanation: Dict,
        feature_names: Optional[List[str]] = None,
        node_names: Optional[Dict[int, str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Generate human-readable text report
        
        Args:
            explanation: Explanation dictionary
            feature_names: List of feature names
            node_names: Dictionary mapping node indices to names
            top_k: Number of top items to include
            
        Returns:
            Text report string
        """
        report = []
        
        node_idx = explanation.get('node_idx', 'Unknown')
        report.append(f"Explanation Report for Node {node_idx}")
        report.append("=" * 50)
        
        # original prediction
        if 'original_prediction' in explanation:
            pred = explanation['original_prediction']
            report.append(f"Original Prediction: {pred:.4f}")
            report.append(f"Predicted Class: {'Anomaly' if pred > 0.5 else 'Normal'}")
            report.append("")
        
        # feature explanation
        if 'feature_shapley_values' in explanation:
            report.append("Top Feature Contributions:")
            report.append("-" * 30)
            
            shapley_values = explanation['feature_shapley_values']
            sorted_items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for i, (feat_idx, value) in enumerate(sorted_items[:top_k]):
                feat_name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"Feature_{feat_idx}"
                contribution = "positive" if value > 0 else "negative"
                report.append(f"{i+1}. {feat_name}: {value:.4f} ({contribution})")
            
            report.append("")
        
        # structure explanation
        if 'structure_shapley_values' in explanation:
            shapley_values = explanation['structure_shapley_values']
            if shapley_values:
                report.append("Top Neighbor Influences:")
                report.append("-" * 30)
                
                sorted_items = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
                
                for i, (node_idx, value) in enumerate(sorted_items[:top_k]):
                    node_name = node_names.get(node_idx, f"Node_{node_idx}") if node_names else f"Node_{node_idx}"
                    influence = "positive" if value > 0 else "negative"
                    report.append(f"{i+1}. {node_name}: {value:.4f} ({influence})")
                
                report.append("")
        
        # comprehensive explanation
        if explanation.get('explanation_type') == 'comprehensive':
            report.append("Comprehensive Analysis:")
            report.append("-" * 30)
            
            feature_exp = explanation.get('feature_explanation', {})
            structure_exp = explanation.get('structure_explanation', {})
            
            # feature summary
            if 'feature_shapley_values' in feature_exp:
                feat_values = list(feature_exp['feature_shapley_values'].values())
                if feat_values:
                    report.append(f"Feature contribution range: [{min(feat_values):.4f}, {max(feat_values):.4f}]")
                    report.append(f"Mean absolute feature contribution: {np.mean(np.abs(feat_values)):.4f}")
            
            # structure summary
            if 'structure_shapley_values' in structure_exp:
                struct_values = list(structure_exp['structure_shapley_values'].values())
                if struct_values:
                    report.append(f"Neighbor influence range: [{min(struct_values):.4f}, {max(struct_values):.4f}]")
                    report.append(f"Mean absolute neighbor influence: {np.mean(np.abs(struct_values)):.4f}")
                    report.append(f"Number of influential neighbors: {len(struct_values)}")
        
        return "\n".join(report)
    
    def save_json_report(
        self,
        explanation: Dict,
        save_path: str,
        feature_names: Optional[List[str]] = None,
        node_names: Optional[Dict[int, str]] = None
    ):
        """
        Save explanation as JSON file
        
        Args:
            explanation: Explanation dictionary
            save_path: Path to save JSON file
            feature_names: List of feature names
            node_names: Dictionary mapping node indices to names
        """
        # create JSON-serializable version
        json_explanation = self._make_json_serializable(explanation.copy())
        
        # add metadata
        json_explanation['metadata'] = {
            'feature_names': feature_names,
            'node_names': node_names,
            'report_generated_at': pd.Timestamp.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(json_explanation, f, indent=2)
        
        self.logger.info(f"Saved JSON explanation to {save_path}")
    
    def save_csv_summary(
        self,
        explanations: Dict[int, Dict],
        save_path: str,
        explanation_type: str = 'feature'
    ):
        """
        Save explanation summary as CSV
        
        Args:
            explanations: Dictionary of explanations indexed by node
            save_path: Path to save CSV file
            explanation_type: Type of explanation to summarize
        """
        data = []
        
        for node_idx, explanation in explanations.items():
            if 'error' in explanation:
                continue
            
            row = {
                'node_idx': node_idx,
                'original_prediction': explanation.get('original_prediction', None)
            }
            
            if explanation_type == 'feature' and 'feature_shapley_values' in explanation:
                shapley_values = explanation['feature_shapley_values']
                for feat_idx, value in shapley_values.items():
                    row[f'feature_{feat_idx}_shapley'] = value
            
            elif explanation_type == 'structure' and 'structure_shapley_values' in explanation:
                shapley_values = explanation['structure_shapley_values']
                for neighbor_idx, value in shapley_values.items():
                    row[f'neighbor_{neighbor_idx}_shapley'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        
        self.logger.info(f"Saved CSV summary to {save_path}")
    
    def _make_json_serializable(self, obj):
        """Convert torch tensors and numpy arrays to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj