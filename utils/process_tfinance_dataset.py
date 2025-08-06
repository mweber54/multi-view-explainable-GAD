#!/usr/bin/env python3
"""
Process T-Finance Dataset to PyTorch Geometric format
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class TFinanceDatasetProcessor:
    """Process the T-Finance dataset from CSV format"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.data_dir = Path("data/processed/tfinance_processed")
        
    def load_csv_data(self):
        """Load CSV data files"""
        self.logger.info("Loading T-Finance CSV data...")
        
        # load node features
        features_path = self.data_dir / "node_features.csv"
        features_df = pd.read_csv(features_path, header=None)
        
        # load node labels
        labels_path = self.data_dir / "node_labels.csv"
        labels_df = pd.read_csv(labels_path, header=None)
        
        # load edges
        edges_path = self.data_dir / "edges.csv"
        edges_df = pd.read_csv(edges_path)
        
        self.logger.info(f"Loaded data:")
        self.logger.info(f"  Features shape: {features_df.shape}")
        self.logger.info(f"  Labels shape: {labels_df.shape}")
        self.logger.info(f"  Edges shape: {edges_df.shape}")
        
        return features_df, labels_df, edges_df
    
    def process_features(self, features_df):
        """Process node features"""
        self.logger.info("Processing node features...")
        
        # remove node ID column (first column)
        features = features_df.iloc[:, 1:].values.astype(np.float32)
        
        self.logger.info(f"Features processed: {features.shape}")
        self.logger.info(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
        
        return features
    
    def process_labels(self, labels_df):
        """Process node labels"""
        self.logger.info("Processing node labels...")
        
        # extract labels (second column)
        labels = labels_df.iloc[:, 1].values.astype(np.int64)
        
        anomaly_count = (labels == 1).sum()
        total_count = len(labels)
        anomaly_ratio = anomaly_count / total_count * 100
        
        self.logger.info(f"Labels processed: {len(labels)} nodes")
        self.logger.info(f"Anomalies: {anomaly_count} ({anomaly_ratio:.2f}%)")
        
        return labels
    
    def process_edges(self, edges_df):
        """Process edge list"""
        self.logger.info("Processing edges...")
        
        # extract edge list (assuming columns are txId1, txId2)
        edges = edges_df[['txId1', 'txId2']].values.astype(np.int64)
        
        # remove self-loops
        mask = edges[:, 0] != edges[:, 1]
        edges = edges[mask]
        
        # convert to bidirectional edges
        edges_bidirectional = np.vstack([edges, edges[:, [1, 0]]])
        
        # remove duplicates
        edges_unique = np.unique(edges_bidirectional, axis=0)
        
        self.logger.info(f"Edges processed:")
        self.logger.info(f"  Original edges: {len(edges_df):,}")
        self.logger.info(f"  After removing self-loops: {len(edges):,}")
        self.logger.info(f"  Final bidirectional edges: {len(edges_unique):,}")
        
        return edges_unique.T
    
    def create_pytorch_geometric_data(self, features, labels, edges):
        """Create PyTorch Geometric data object"""
        self.logger.info("Creating PyTorch Geometric data...")
        
        # normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # convert to tensors
        x = torch.tensor(features_normalized, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        
        # create train/val/test masks
        num_nodes = len(labels)
        labeled_indices = np.arange(num_nodes)
        
        # stratified split
        train_indices, temp_indices = train_test_split(
            labeled_indices, test_size=0.4, random_state=42, stratify=labels
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.75, random_state=42, stratify=labels[temp_indices]
        )
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        # create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        # add metadata
        data.num_nodes = num_nodes
        data.num_edges = edge_index.shape[1]
        data.num_features = features.shape[1]
        data.num_classes = 2
        
        # calculate statistics
        anomaly_count = (labels == 1).sum()
        anomaly_ratio = anomaly_count / num_nodes * 100
        
        self.logger.info(f"Created PyTorch Geometric data:")
        self.logger.info(f"  Nodes: {data.num_nodes:,}")
        self.logger.info(f"  Edges: {data.num_edges:,}")
        self.logger.info(f"  Features: {data.num_features}")
        self.logger.info(f"  Anomalies: {anomaly_count} ({anomaly_ratio:.2f}%)")
        self.logger.info(f"  Train: {train_mask.sum()}")
        self.logger.info(f"  Val: {val_mask.sum()}")
        self.logger.info(f"  Test: {test_mask.sum()}")
        
        return data
    
    def save_processed_data(self, data):
        """Save processed data"""
        output_path = Path("data/processed/tfinance_static.pt")
        output_path.parent.mkdir(exist_ok=True)
        
        torch.save(data, output_path)
        self.logger.info(f"Saved processed data to: {output_path}")
        
        # save masks separately for compatibility
        mask_dir = Path("data/processed/tfinance_processed")
        mask_dir.mkdir(exist_ok=True)
        
        train_indices = torch.where(data.train_mask)[0].numpy()
        val_indices = torch.where(data.val_mask)[0].numpy()
        test_indices = torch.where(data.test_mask)[0].numpy()
        
        np.savetxt(mask_dir / "train_mask.csv", train_indices, fmt='%d', delimiter=',')
        np.savetxt(mask_dir / "val_mask.csv", val_indices, fmt='%d', delimiter=',')
        np.savetxt(mask_dir / "test_mask.csv", test_indices, fmt='%d', delimiter=',')
        
        self.logger.info(f"Saved masks to: {mask_dir}")
        
        return output_path
    
    def process(self):
        """Main processing function"""
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING T-FINANCE DATASET")
        self.logger.info("=" * 60)
        
        try:
            # load CSV data
            features_df, labels_df, edges_df = self.load_csv_data()
            
            # process components
            features = self.process_features(features_df)
            labels = self.process_labels(labels_df)
            edges = self.process_edges(edges_df)
            
            # create PyTorch Geometric data
            data = self.create_pytorch_geometric_data(features, labels, edges)
            
            # save processed data
            output_path = self.save_processed_data(data)
            
            self.logger.info("=" * 60)
            self.logger.info("T-FINANCE PROCESSING COMPLETED SUCCESSFULLY")
            self.logger.info(f"Output: {output_path}")
            self.logger.info("=" * 60)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

def main():
    """Main function"""
    processor = TFinanceDatasetProcessor()
    return processor.process()

if __name__ == "__main__":
    result = main()