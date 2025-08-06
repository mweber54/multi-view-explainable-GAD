#!/usr/bin/env python3
"""
Process Yelp Dataset to PyTorch Geometric format
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

class YelpDatasetProcessor:
    """Process the Yelp dataset from CSV format"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.data_dir = Path("data/processed/yelp_processed")
        
    def load_csv_data(self):
        """Load CSV data files"""
        self.logger.info("Loading Yelp CSV data...")
        
        # load node features
        features_path = self.data_dir / "node_features.csv"
        features_df = pd.read_csv(features_path, header=None)
        
        # load node labels
        labels_path = self.data_dir / "node_labels.csv"
        labels_df = pd.read_csv(labels_path, header=None)
        
        # get edges path for chunked processing
        edges_path = self.data_dir / "edges.csv"
        
        self.logger.info(f"Loaded features and labels:")
        self.logger.info(f"  Features shape: {features_df.shape}")
        self.logger.info(f"  Labels shape: {labels_df.shape}")
        
        return features_df, labels_df, edges_path
    
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
    
    def process_edges_efficient(self, edges_path, num_nodes):
        """Process edge list efficiently with chunking"""
        self.logger.info("Processing edges efficiently...")
        
        edges_list = []
        chunk_size = 1000000  # process 1M edges at a time
        total_processed = 0
        
        # read edges in chunks, skip header
        first_chunk = True
        for chunk in pd.read_csv(edges_path, chunksize=chunk_size):
            # skip header in first chunk, extract edge columns
            if first_chunk and 'txId1' in chunk.columns:
                edges_chunk = chunk[['txId1', 'txId2']].values.astype(np.int64)
                first_chunk = False
            else:
                edges_chunk = chunk.values.astype(np.int64)
            
            # filter valid edges (within node range)
            valid_mask = (edges_chunk[:, 0] < num_nodes) & (edges_chunk[:, 1] < num_nodes)
            edges_chunk = edges_chunk[valid_mask]
            
            # remove self-loops
            non_self_loop_mask = edges_chunk[:, 0] != edges_chunk[:, 1]
            edges_chunk = edges_chunk[non_self_loop_mask]
            
            if len(edges_chunk) > 0:
                edges_list.append(edges_chunk)
            
            total_processed += len(edges_chunk)
            self.logger.info(f"Processed chunk with {len(edges_chunk):,} valid edges (total: {total_processed:,})")
            
            # memory management - limit to reasonable size
            if total_processed > 2000000:  # Stop at 2M edges
                self.logger.info(f"Reached edge limit of 2M edges, stopping processing")
                break
        
        if not edges_list:
            self.logger.warning("No valid edges found, creating minimal connectivity")
            # create a simple ring graph as fallback
            edges = np.array([[i, (i + 1) % num_nodes] for i in range(min(1000, num_nodes))]).T
        else:
            # combine all chunks
            edges = np.vstack(edges_list)
            
            # sample edges if still too many
            max_edges = 1500000  # Final limit
            if len(edges) > max_edges:
                self.logger.info(f"Sampling {max_edges:,} edges from {len(edges):,} total")
                sample_indices = np.random.choice(len(edges), max_edges, replace=False)
                edges = edges[sample_indices]
            
            # convert to bidirectional
            edges_bidirectional = np.vstack([edges, edges[:, [1, 0]]])
            
            # remove duplicates efficiently
            edges_unique = np.unique(edges_bidirectional, axis=0)
            edges = edges_unique.T
        
        self.logger.info(f"Final edge processing complete:")
        self.logger.info(f"  Final edges: {edges.shape[1]:,}")
        
        return edges
    
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
        output_path = Path("data/processed/yelp_static.pt")
        output_path.parent.mkdir(exist_ok=True)
        
        torch.save(data, output_path)
        self.logger.info(f"Saved processed data to: {output_path}")
        
        return output_path
    
    def process(self):
        """Main processing function"""
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING YELP DATASET")
        self.logger.info("=" * 60)
        
        try:
            # load CSV data
            features_df, labels_df, edges_path = self.load_csv_data()
            
            # process components
            features = self.process_features(features_df)
            labels = self.process_labels(labels_df)
            edges = self.process_edges_efficient(edges_path, len(labels))
            
            # create PyTorch Geometric data
            data = self.create_pytorch_geometric_data(features, labels, edges)
            
            # save processed data
            output_path = self.save_processed_data(data)
            
            self.logger.info("=" * 60)
            self.logger.info("YELP PROCESSING COMPLETED SUCCESSFULLY")
            self.logger.info(f"Output: {output_path}")
            self.logger.info("=" * 60)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

def main():
    """Main function"""
    processor = YelpDatasetProcessor()
    return processor.process()

if __name__ == "__main__":
    result = main()