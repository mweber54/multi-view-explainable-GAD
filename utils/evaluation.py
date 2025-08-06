import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time
from contextlib import contextmanager

@contextmanager
def timer(logger: logging.Logger, description: str):
    """Simple timer context manager"""
    start_time = time.time()
    logger.info(f"Starting {description}...")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed {description} in {elapsed:.2f} seconds")


class AnomalyEvaluator:
    """Comprehensive evaluation for anomaly detection"""
    
    def __init__(self, logger: logging.Logger, save_dir: Optional[str] = None):
        self.logger = logger
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_threshold_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                threshold: float) -> Dict[str, float]:
        """Compute metrics at a specific threshold"""
        y_pred = (y_scores >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # compute various metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, 
                             metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold based on specified metric"""
        
        # generate candidate thresholds
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
        
        best_threshold = None
        best_score = -1
        best_metrics = None
        
        for threshold in thresholds:
            metrics = self.compute_threshold_metrics(y_true, y_scores, threshold)
            
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_threshold = threshold
                best_metrics = metrics
        
        self.logger.info(f"Optimal threshold ({metric}): {best_threshold:.4f} -> {best_score:.4f}")
        return best_threshold, best_metrics
    
    def compute_comprehensive_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        
        with timer(self.logger, "comprehensive metrics computation"):
            metrics = {}
            
            # threshold-independent metrics
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['average_precision'] = average_precision_score(y_true, y_scores)
            
            # find optimal thresholds for different metrics
            f1_threshold, f1_metrics = self.find_optimal_threshold(y_true, y_scores, 'f1_score')
            ba_threshold, ba_metrics = self.find_optimal_threshold(y_true, y_scores, 'balanced_accuracy')
            
            # add threshold-dependent metrics
            metrics.update({f'f1_optimal_{k}': v for k, v in f1_metrics.items()})
            metrics.update({f'ba_optimal_{k}': v for k, v in ba_metrics.items()})
            
            # statistical summaries
            metrics['anomaly_ratio'] = np.mean(y_true)
            metrics['score_mean'] = np.mean(y_scores)
            metrics['score_std'] = np.std(y_scores)
            metrics['score_min'] = np.min(y_scores)
            metrics['score_max'] = np.max(y_scores)
            metrics['num_samples'] = len(y_true)
            
            return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      title: str = "ROC Curve") -> None:
        """Plot ROC curve"""
        if self.save_dir is None:
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = self.save_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved ROC curve: {save_path}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> None:
        """Plot Precision-Recall curve"""
        if self.save_dir is None:
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
        plt.axhline(y=np.mean(y_true), color='k', linestyle='--', 
                   linewidth=1, label=f'Random (AP = {np.mean(y_true):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        save_path = self.save_dir / "precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved PR curve: {save_path}")
    
    def plot_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray,
                               title: str = "Score Distribution") -> None:
        """Plot score distribution by class"""
        if self.save_dir is None:
            return
        
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.save_dir / "score_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved score distribution: {save_path}")
    
    def save_metrics_report(self, metrics: Dict[str, float], filename: str = "metrics_report.json") -> None:
        """Save metrics to JSON file"""
        if self.save_dir is None:
            return
        
        import json
        
        # convert numpy types to python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics report: {save_path}")
    
    def evaluate_model_predictions(self, y_true: np.ndarray, y_scores: np.ndarray,
                                 model_name: str = "Model") -> Dict[str, float]:
        """Complete evaluation pipeline"""
        
        self.logger.info(f"Evaluating {model_name}...")
        
        # validate inputs
        if len(y_true) != len(y_scores):
            raise ValueError("y_true and y_scores must have the same length")
        
        if len(np.unique(y_true)) != 2:
            raise ValueError("y_true must be binary (0/1)")
        
        # compute metrics
        metrics = self.compute_comprehensive_metrics(y_true, y_scores)
        
        # generate plots
        self.plot_roc_curve(y_true, y_scores, f"{model_name} - ROC Curve")
        self.plot_precision_recall_curve(y_true, y_scores, f"{model_name} - PR Curve")
        self.plot_score_distribution(y_true, y_scores, f"{model_name} - Score Distribution")
        
        # save report
        self.save_metrics_report(metrics, f"{model_name.lower()}_metrics.json")
        
        # log key metrics
        self.logger.info(f"{model_name} Evaluation Results:")
        key_metrics = ['roc_auc', 'average_precision', 'f1_optimal_f1_score', 
                      'ba_optimal_balanced_accuracy', 'anomaly_ratio']
        
        for metric in key_metrics:
            if metric in metrics:
                self.logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        return metrics


def evaluate_embeddings_multiple_methods(embeddings: torch.Tensor, y_true: np.ndarray, 
                                       logger: logging.Logger, save_dir: str = "evaluation_results") -> Dict[str, Dict[str, float]]:
    """Evaluate embeddings using multiple anomaly scoring methods"""
    
    evaluator = AnomalyEvaluator(logger, save_dir)
    results = {}
    
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Method 1: L2 norm
    l2_scores = np.linalg.norm(embeddings_np, axis=1)
    results['l2_norm'] = evaluator.evaluate_model_predictions(y_true, l2_scores, "L2_Norm")
    
    # Method 2: Mahalanobis distance (if enough normal samples)
    normal_embeddings = embeddings_np[y_true == 0]
    if len(normal_embeddings) > embeddings_np.shape[1]:  # Need more samples than dimensions
        try:
            mean = np.mean(normal_embeddings, axis=0)
            cov = np.cov(normal_embeddings.T)
            cov_inv = np.linalg.pinv(cov)
            
            mahal_scores = []
            for emb in embeddings_np:
                diff = emb - mean
                mahal_scores.append(np.sqrt(diff.T @ cov_inv @ diff))
            
            results['mahalanobis'] = evaluator.evaluate_model_predictions(
                y_true, np.array(mahal_scores), "Mahalanobis"
            )
        except Exception as e:
            logger.warning(f"Mahalanobis distance computation failed: {e}")
    
    # Method 3: Distance to k-nearest normal neighbors
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(n_neighbors=min(10, len(normal_embeddings)), metric='euclidean')
    knn.fit(normal_embeddings)
    
    distances, _ = knn.kneighbors(embeddings_np)
    knn_scores = np.mean(distances, axis=1)
    results['knn_distance'] = evaluator.evaluate_model_predictions(y_true, knn_scores, "KNN_Distance")
    
    # log comparison
    logger.info("Method Comparison:")
    for method, metrics in results.items():
        logger.info(f"  {method}: AUC={metrics['roc_auc']:.4f}, AP={metrics['average_precision']:.4f}")
    
    return results