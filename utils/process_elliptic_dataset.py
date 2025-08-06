#!/usr/bin/env python3
"""
Elliptic Bitcoin Transaction Network Dataset Processor

This script processes the Elliptic Bitcoin dataset CSV files and creates a PyTorch Geometric
compatible graph dataset for our GAD pipeline.

Dataset Details:
- Elliptic Bitcoin transaction network for money laundering detection
- Classes: 1 (illicit), 2 (licit), 'unknown' (unlabeled)
- Features: 166 features per transaction (93 local + 73 aggregated)
- Goal: Beat 66% ROC-AUC baseline
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

def setup_logging():
    """Setup logging for dataset processing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_elliptic_data(data_dir="elliptic"):
    """Load all Elliptic dataset CSV files"""
    logger = setup_logging()
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Elliptic data directory {data_dir} not found")
    
    logger.info("Loading Elliptic Bitcoin dataset...")
    
    # load node features
    features_file = data_path / "elliptic_txs_features.csv"
    logger.info(f"Loading features from {features_file}")
    features_df = pd.read_csv(features_file, header=None)
    
    # load node classes/labels
    classes_file = data_path / "elliptic_txs_classes.csv"
    logger.info(f"Loading classes from {classes_file}")
    classes_df = pd.read_csv(classes_file)
    
    # load edge list
    edges_file = data_path / "elliptic_txs_edgelist.csv"
    logger.info(f"Loading edges from {edges_file}")
    edges_df = pd.read_csv(edges_file)
    
    logger.info(f"Dataset loaded:")
    logger.info(f"  Features: {features_df.shape}")
    logger.info(f"  Classes: {classes_df.shape}")
    logger.info(f"  Edges: {edges_df.shape}")
    
    return features_df, classes_df, edges_df

def process_node_features(features_df):
    """Process node features"""
    logger = setup_logging()
    
    # first column is txId, rest are features
    node_ids = features_df.iloc[:, 0].values
    features = features_df.iloc[:, 1:].values.astype(np.float32)
    
    logger.info(f"Processing {len(node_ids)} nodes with {features.shape[1]} features")
    
    # create node ID to index mapping
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # standardize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    logger.info(f"Features normalized: mean={features_normalized.mean():.4f}, std={features_normalized.std():.4f}")
    
    return node_ids, features_normalized, id_to_idx

def process_node_labels(classes_df, id_to_idx):
    """Process node labels"""
    logger = setup_logging()
    
    # map string labels to integers
    # 1 (illicit) -> 1, 2 (licit) -> 0, 'unknown' -> -1
    label_mapping = {'1': 1, '2': 0, 'unknown': -1}
    
    labels = np.full(len(id_to_idx), -1, dtype=np.int64)  # Default to unknown
    
    labeled_count = 0
    illicit_count = 0
    licit_count = 0
    
    for _, row in classes_df.iterrows():
        tx_id = row['txId']
        class_label = str(row['class'])
        
        if tx_id in id_to_idx:
            idx = id_to_idx[tx_id]
            labels[idx] = label_mapping[class_label]
            
            if class_label != 'unknown':
                labeled_count += 1
                if class_label == '1':
                    illicit_count += 1
                else:
                    licit_count += 1
    
    logger.info(f"Label distribution:")
    logger.info(f"  Total nodes: {len(labels)}")
    logger.info(f"  Labeled nodes: {labeled_count}")
    logger.info(f"  Illicit (1): {illicit_count}")
    logger.info(f"  Licit (0): {licit_count}")
    logger.info(f"  Unknown (-1): {len(labels) - labeled_count}")
    logger.info(f"  Anomaly ratio: {illicit_count / labeled_count * 100:.2f}%")
    
    return labels

def process_edges(edges_df, id_to_idx):
    """Process edge list"""
    logger = setup_logging()
    
    edge_list = []
    missing_nodes = 0
    
    for _, row in edges_df.iterrows():
        src_id = row['txId1']
        dst_id = row['txId2']
        
        if src_id in id_to_idx and dst_id in id_to_idx:
            src_idx = id_to_idx[src_id]
            dst_idx = id_to_idx[dst_id]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])  # Make undirected
        else:
            missing_nodes += 1
    
    edges = np.array(edge_list).T if edge_list else np.array([[], []], dtype=np.int64)
    
    logger.info(f"Processed edges:")
    logger.info(f"  Original edges: {len(edges_df)}")
    logger.info(f"  Missing nodes: {missing_nodes}")
    logger.info(f"  Final edges (undirected): {edges.shape[1]}")
    
    return edges

def create_train_val_test_masks(labels, test_size=0.3, val_size=0.15, random_state=42):
    """Create train/validation/test masks for labeled nodes"""
    logger = setup_logging()
    
    # get indices of labeled nodes only
    labeled_indices = np.where(labels >= 0)[0]
    labeled_labels = labels[labeled_indices]
    
    logger.info(f"Creating masks for {len(labeled_indices)} labeled nodes")
    
    # first split: train+val vs test
    train_val_indices, test_indices, _, _ = train_test_split(
        labeled_indices, labeled_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labeled_labels
    )
    
    # second split: train vs val
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, labels[train_val_indices],
        test_size=val_size / (1 - test_size),  # Adjust for remaining data
        random_state=random_state,
        stratify=labels[train_val_indices]
    )
    
    # create boolean masks
    total_nodes = len(labels)
    train_mask = np.zeros(total_nodes, dtype=bool)
    val_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    logger.info(f"Mask distribution:")
    logger.info(f"  Train: {train_mask.sum()} ({train_mask.sum()/labeled_indices.shape[0]*100:.1f}%)")
    logger.info(f"  Val: {val_mask.sum()} ({val_mask.sum()/labeled_indices.shape[0]*100:.1f}%)")
    logger.info(f"  Test: {test_mask.sum()} ({test_mask.sum()/labeled_indices.shape[0]*100:.1f}%)")
    
    # check class balance in each split
    for split_name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        split_labels = labels[mask]
        illicit_ratio = (split_labels == 1).sum() / len(split_labels) * 100
        logger.info(f"  {split_name} illicit ratio: {illicit_ratio:.1f}%")
    
    return train_mask, val_mask, test_mask

def create_pytorch_geometric_data(features, labels, edges, train_mask, val_mask, test_mask):
    """Create PyTorch Geometric Data object"""
    logger = setup_logging()
    
    # convert to tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long)
    
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)
    
    # create Data object
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask_tensor,
        val_mask=val_mask_tensor,
        test_mask=test_mask_tensor
    )
    
    # add metadata
    data.num_nodes = len(features)
    data.num_edges = edges.shape[1]
    data.num_features = features.shape[1]
    data.num_classes = 2  # Binary: licit vs illicit
    
    logger.info(f"Created PyTorch Geometric data:")
    logger.info(f"  Nodes: {data.num_nodes}")
    logger.info(f"  Edges: {data.num_edges}")
    logger.info(f"  Features: {data.num_features}")
    logger.info(f"  Classes: {data.num_classes}")
    
    return data

def save_processed_data(data, output_file="elliptic_static.pt"):
    """Save processed data to file"""
    logger = setup_logging()
    
    output_path = Path(output_file)
    torch.save(data, output_path)
    
    logger.info(f"Saved processed Elliptic dataset to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

def save_mask_files(train_mask, val_mask, test_mask, output_dir="elliptic_processed"):
    """Save mask files for compatibility with existing pipeline"""
    logger = setup_logging()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # get indices where masks are True
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # save as CSV files
    pd.DataFrame({'node_idx': train_indices}).to_csv(output_path / 'train_mask.csv', index=False)
    pd.DataFrame({'node_idx': val_indices}).to_csv(output_path / 'val_mask.csv', index=False)
    pd.DataFrame({'node_idx': test_indices}).to_csv(output_path / 'test_mask.csv', index=False)
    
    logger.info(f"Saved mask files to {output_path}")

def main():
    """Main processing function"""
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("ELLIPTIC BITCOIN DATASET PROCESSING")
    logger.info("="*60)
    
    try:
        # load raw data
        features_df, classes_df, edges_df = load_elliptic_data()
        
        # process features
        node_ids, features, id_to_idx = process_node_features(features_df)
        
        # process labels
        labels = process_node_labels(classes_df, id_to_idx)
        
        # process edges
        edges = process_edges(edges_df, id_to_idx)
        
        # create masks
        train_mask, val_mask, test_mask = create_train_val_test_masks(labels)
        
        # create PyTorch Geometric data
        data = create_pytorch_geometric_data(features, labels, edges, train_mask, val_mask, test_mask)
        
        # save data
        save_processed_data(data)
        save_mask_files(train_mask, val_mask, test_mask)
        
        logger.info("="*60)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # dataset summary
        labeled_nodes = (labels >= 0).sum()
        illicit_nodes = (labels == 1).sum()
        
        logger.info("DATASET SUMMARY:")
        logger.info(f"  Total nodes: {len(labels):,}")
        logger.info(f"  Total edges: {edges.shape[1]:,}")
        logger.info(f"  Features per node: {features.shape[1]}")
        logger.info(f"  Labeled nodes: {labeled_nodes:,}")
        logger.info(f"  Illicit transactions: {illicit_nodes:,}")
        logger.info(f"  Anomaly ratio: {illicit_nodes/labeled_nodes*100:.2f}%")
        logger.info("")
        logger.info("FILES CREATED:")
        logger.info("  - elliptic_static.pt (main graph file)")
        logger.info("  - elliptic_processed/train_mask.csv")
        logger.info("  - elliptic_processed/val_mask.csv")
        logger.info("  - elliptic_processed/test_mask.csv")
        logger.info("")
        logger.info("READY FOR GAD PIPELINE EVALUATION!")
        logger.info("Target: Beat 66% ROC-AUC baseline")
        
        return data
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    processed_data = main()