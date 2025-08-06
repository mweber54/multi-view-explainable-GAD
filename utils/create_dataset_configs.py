#!/usr/bin/env python3
"""
Create dataset-specific configurations for all processed datasets
"""

import json
from pathlib import Path

# dataset information from processing results
DATASETS_INFO = {
    'amazon': {
        'nodes': 11944, 'edges': 8847096, 'features': 25, 'anomaly_ratio': 0.069,
        'processed_dir': 'amazon_processed'
    },
    'yelp': {
        'nodes': 45954, 'edges': 7739912, 'features': 32, 'anomaly_ratio': 0.145,
        'processed_dir': 'yelp_processed'  
    },
    'weibo': {
        'nodes': 8405, 'edges': 416368, 'features': 400, 'anomaly_ratio': 0.103,
        'processed_dir': 'weibo_processed'
    },
    'tsocial': {
        'nodes': 5781065, 'edges': 151992081, 'features': 10, 'anomaly_ratio': 0.030,
        'processed_dir': 'tsocial_processed'
    },
    'tfinance': {
        'nodes': 39357, 'edges': 42484443, 'features': 10, 'anomaly_ratio': 0.046,
        'processed_dir': 'tfinance_processed'
    },
    'tolokers': {
        'nodes': 11758, 'edges': 530758, 'features': 10, 'anomaly_ratio': 0.218,
        'processed_dir': 'tolokers_processed'
    },
    'reddit': {
        'nodes': 10984, 'edges': 168016, 'features': 64, 'anomaly_ratio': 0.033,
        'processed_dir': 'reddit_processed'
    }
}

def create_config_for_dataset(dataset_name: str, dataset_info: dict) -> dict:
    """Create a configuration tailored for a specific dataset"""
    
    # base configuration
    config = {
        "data": {
            "features_path": f"{dataset_info['processed_dir']}/node_features.csv",
            "edges_path": f"{dataset_info['processed_dir']}/edges.csv", 
            "labels_path": f"{dataset_info['processed_dir']}/node_labels.csv",
            "static_save_path": f"{dataset_name}_static.pt",
            "embeddings_path": f"{dataset_name}_embeddings.npy",
            "standardize_features": True,
            "include_centrality": True,  
            "include_motifs": True,      
            "include_temporal": True,    
            "betweenness_sample_k": 100
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "temperature": 0.5,
            "edge_dropout_prob": 0.2,
            "feature_mask_prob": 0.2,
            "gradient_clip_norm": 1.0,
            "save_checkpoints": True,
            "checkpoint_dir": f"checkpoints_{dataset_name}",
            "save_every_n_epochs": 10
        },
        "system": {
            "device": "auto",
            "random_seed": 42,
            "num_workers": 0,
            "log_level": "INFO", 
            "log_file": f"{dataset_name}_pipeline.log"
        }
    }
    
    # adjust parameters based on dataset size and characteristics
    nodes = dataset_info['nodes']
    edges = dataset_info['edges']
    features = dataset_info['features']
    
    # architecture scaling
    if nodes < 20000: 
        config["model"] = {
            "hidden_channels": 128,
            "latent_dim": 64,
            "projection_dim": 64,
            "dropout": 0.3,
            "num_heads": 2,
            "classifier_hidden": 64
        }
        config["training"].update({
            "epochs_pretrain": 20,
            "epochs_finetune": 20,
            "pretrain_batch_size": 1024,
            "finetune_batch_size": 2048,
            "eval_batch_size": 2048,
            "num_neighbors": [15, 10]
        })
        # enable more features for small datasets
        config["data"]["include_centrality"] = True
        config["data"]["include_motifs"] = True
        
    elif nodes < 100000: 
        config["model"] = {
            "hidden_channels": 64,
            "latent_dim": 32,
            "projection_dim": 32,
            "dropout": 0.3,
            "num_heads": 2,
            "classifier_hidden": 32
        }
        config["training"].update({
            "epochs_pretrain": 15,
            "epochs_finetune": 15,
            "pretrain_batch_size": 512,
            "finetune_batch_size": 1024,
            "eval_batch_size": 1024,
            "num_neighbors": [10, 5]
        })
        
    else:  # large datasets  
        config["model"] = {
            "hidden_channels": 32,
            "latent_dim": 16,
            "projection_dim": 16,
            "dropout": 0.2,
            "num_heads": 1,
            "classifier_hidden": 16
        }
        config["training"].update({
            "epochs_pretrain": 5,
            "epochs_finetune": 5,
            "pretrain_batch_size": 256,
            "finetune_batch_size": 512,
            "eval_batch_size": 512,
            "num_neighbors": [5, 5]
        })
        
    # adjustments for high-feature datasets
    if features > 100:
        config["model"]["hidden_channels"] = min(config["model"]["hidden_channels"], features // 2)
        config["data"]["include_motifs"] = False  # motifs are too computationally expensive on big data
        
    return config

def main():
    """Create configurations for all datasets"""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    print("Creating dataset-specific configurations...")
    
    for dataset_name, dataset_info in DATASETS_INFO.items():
        print(f"Creating config for {dataset_name}...")
        
        config = create_config_for_dataset(dataset_name, dataset_info)
        
        # save configuration
        config_path = configs_dir / f"{dataset_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"  Saved: {config_path}")
        print(f"  Model: {config['model']['latent_dim']}D latent, "
              f"{config['training']['epochs_pretrain']} pretrain epochs")
    
    # script to run all datasets
    script_content = '''#!/usr/bin/env python3
"""
Run GAD pipeline on all processed datasets
"""

import subprocess
import sys
from pathlib import Path

DATASETS = ['amazon', 'yelp', 'weibo', 'tsocial', 'tfinance', 'tolokers', 'reddit']

def run_dataset(dataset_name):
    """Run pipeline on a single dataset"""
    print(f"\\n{'='*60}")
    print(f"Running GAD pipeline on {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    config_path = f"configs/{dataset_name}_config.json"
    output_dir = f"results_{dataset_name}"
    
    cmd = [
        sys.executable, "main.py",
        "--mode", "full",
        "--config", config_path,
        "--output-dir", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"{dataset_name} completed successfully")
            return True
        else:
            print(f"{dataset_name} failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{dataset_name} timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"{dataset_name} crashed: {e}")
        return False

def main():
    """Run pipeline on all datasets"""
    successful = []
    failed = []
    
    for dataset_name in DATASETS:
        if run_dataset(dataset_name):
            successful.append(dataset_name)
        else:
            failed.append(dataset_name)
    
    # Summary
    print(f"\\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Successful ({len(successful)}): {successful}")
    print(f"Failed ({len(failed)}): {failed}")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("run_all_datasets.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\nCreated {len(DATASETS_INFO)} configurations")
    print(f"Saved master script: {script_path}")
    
    return list(DATASETS_INFO.keys())

if __name__ == "__main__":
    datasets = main()
    print(f"\\nReady to run GAD pipeline on {len(datasets)} datasets!")
    print("Usage: python run_all_datasets.py")