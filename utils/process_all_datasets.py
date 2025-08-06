#!/usr/bin/env python3
"""
Process all graph datasets into CSV format for the GAD pipeline
This script processes multiple DGL graph files and extracts their data into CSV format
"""

import dgl
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from typing import List, Dict, Any
import traceback

# logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASETS = [
    'amazon',
    'yelp', 
    'weibo',
    'tsocial',
    'tfinance', 
    'tolokers',
    'reddit'
]

def process_single_dataset(dataset_name: str, base_dir: Path) -> Dict[str, Any]:
    """Process a single dataset and extract CSV files"""
    logger.info(f"Processing dataset: {dataset_name}")
    
    dataset_dir = base_dir / f"{dataset_name}_processed"
    dataset_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # load graph
        logger.info(f"Loading graph file: {dataset_name}")
        graphs, _ = dgl.load_graphs(str(base_dir / dataset_name))
        graph = graphs[0]
        
        # get basic statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        logger.info(f"Graph loaded: {num_nodes} nodes, {num_edges} edges")
        
        # extract edges - ensure node IDs match feature indices
        logger.info("Extracting edges...")
        src, dst = graph.edges()
        
        # convert to consecutive node IDs (0 to num_nodes-1)
        src_numpy = src.numpy()
        dst_numpy = dst.numpy()
        
        # create mapping from original IDs to consecutive IDs
        all_node_ids = np.unique(np.concatenate([src_numpy, dst_numpy]))
        if len(all_node_ids) != num_nodes:
            logger.warning(f"Node ID mismatch: {len(all_node_ids)} unique IDs vs {num_nodes} nodes")
            # create a mapping to ensure consistency
            node_id_map = {old_id: new_id for new_id, old_id in enumerate(all_node_ids)}
            src_remapped = np.array([node_id_map.get(x, x) for x in src_numpy])
            dst_remapped = np.array([node_id_map.get(x, x) for x in dst_numpy])
        else:
            src_remapped = src_numpy
            dst_remapped = dst_numpy
            
        edges_df = pd.DataFrame({
            'txId1': src_remapped,
            'txId2': dst_remapped
        })
        edges_path = dataset_dir / 'edges.csv'
        edges_df.to_csv(edges_path, index=False)
        logger.info(f"Saved edges to: {edges_path}")
        
        # extract features
        logger.info("Extracting node features...")
        features = graph.ndata['feature'].numpy()
        
        # create feature dataframe with proper format for our pipeline
        # add tx_id and time_step columns (using consecutive node IDs)
        feature_data = np.column_stack([
            np.arange(num_nodes).astype(str),  # tx_id as string
            np.zeros(num_nodes),   # time_step (dummy)
            features
        ])
        
        features_df = pd.DataFrame(feature_data)
        features_path = dataset_dir / 'node_features.csv'
        features_df.to_csv(features_path, index=False, header=False)  # No header as expected by pipeline
        logger.info(f"Saved features to: {features_path}")
        
        # extract labels
        logger.info("Extracting node labels...")
        labels = graph.ndata['label'].numpy()
        
        # create labels dataframe with proper format
        labels_data = np.column_stack([
            np.arange(num_nodes).astype(str),  # tx_id as string  
            labels
        ])
        
        labels_df = pd.DataFrame(labels_data)
        labels_path = dataset_dir / 'node_labels.csv'
        labels_df.to_csv(labels_path, index=False, header=False)  # No header as expected
        logger.info(f"Saved labels to: {labels_path}")
        
        # extract masks if available
        mask_info = {}
        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            if mask_key in graph.ndata:
                logger.info(f"Extracting {mask_key}...")
                mask = graph.ndata[mask_key].numpy()
                mask_df = pd.DataFrame({
                    'node_id': range(len(mask)),
                    'mask': mask.astype(int)
                })
                mask_path = dataset_dir / f'{mask_key}.csv'
                mask_df.to_csv(mask_path, index=False)
                mask_info[mask_key] = mask_path
                logger.info(f"Saved {mask_key} to: {mask_path}")
        
        # calculate statistics
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))
        
        processing_time = time.time() - start_time
        
        # summary statistics
        stats = {
            'dataset_name': dataset_name,
            'num_nodes': int(num_nodes),
            'num_edges': int(num_edges),
            'num_features': int(features.shape[1]),
            'label_distribution': label_distribution,
            'processing_time': processing_time,
            'files_created': {
                'edges': str(edges_path),
                'features': str(features_path),
                'labels': str(labels_path),
                'masks': mask_info
            },
            'success': True
        }
        
        logger.info(f"Successfully processed {dataset_name} in {processing_time:.2f} seconds")
        logger.info(f"Label distribution: {label_distribution}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {dataset_name}: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'dataset_name': dataset_name,
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time
        }

def main():
    """Main function to process all datasets"""
    logger.info("Starting multi-dataset processing...")
    
    base_dir = Path(".")
    results = []
    successful_datasets = []
    failed_datasets = []
    
    total_start_time = time.time()
    
    for dataset_name in DATASETS:
        logger.info("="*60)
        
        # check if dataset file exists
        dataset_file = base_dir / dataset_name
        if not dataset_file.exists():
            logger.warning(f"Dataset file not found: {dataset_file}")
            failed_datasets.append(dataset_name)
            continue
        
        # process the dataset
        result = process_single_dataset(dataset_name, base_dir)
        results.append(result)
        
        if result['success']:
            successful_datasets.append(dataset_name)
        else:
            failed_datasets.append(dataset_name)
    
    total_time = time.time() - total_start_time
    
    # print summary
    logger.info("="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Successful datasets ({len(successful_datasets)}): {successful_datasets}")
    if failed_datasets:
        logger.info(f"Failed datasets ({len(failed_datasets)}): {failed_datasets}")
    
    # detailed statistics
    logger.info("\nDataset Statistics:")
    logger.info("-" * 60)
    for result in results:
        if result['success']:
            logger.info(f"{result['dataset_name']:>10}: "
                       f"{result['num_nodes']:>8} nodes, "
                       f"{result['num_edges']:>10} edges, "
                       f"{result['num_features']:>3} features")
    
    # save summary to JSON
    import json
    summary_path = base_dir / 'dataset_processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved processing summary to: {summary_path}")
    
    return successful_datasets, failed_datasets

if __name__ == "__main__":
    successful, failed = main()
    
    if successful:
        print(f"\nSuccessfully processed {len(successful)} datasets: {successful}")
    if failed:
        print(f"\nFailed to process {len(failed)} datasets: {failed}")
    
    print(f"\nReady to run GAD pipeline on processed datasets!")