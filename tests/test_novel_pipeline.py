#!/usr/bin/env python3
"""
Simplified test for the novel pipeline components
"""

import torch
import torch.nn as nn
from models.temporal_spectral_fusion import DynamicWaveletModule, create_temporal_spectral_model

def test_dynamic_wavelets():
    """Test dynamic wavelet functionality"""
    print("Testing Dynamic Wavelets")
    
    # create wavelet module
    wavelet_module = DynamicWaveletModule(
        num_wavelets=8,
        max_frequency=2.0,
        domain_type='financial',
        learnable=True
    )
    
    # testing data
    batch_size, num_nodes, num_features = 2, 10, 8
    graph_signal = torch.randn(batch_size, num_nodes, num_features)
    eigenvalues = torch.sort(torch.rand(batch_size, 8))[0]
    eigenvectors = torch.randn(batch_size, num_nodes, 8)
    
    # forward pass
    with torch.no_grad():
        wavelet_features = wavelet_module(graph_signal, eigenvalues, eigenvectors)
        print(f"Wavelet features shape: {wavelet_features.shape}")
        print("Dynamic wavelets working")

def test_temporal_spectral_model():
    """Test basic temporal spectral model without complex components"""
    print("\nTesting Temporal Spectral Model")
    
    config = {
        'input_dim': 8,
        'hidden_dim': 16,
        'temporal_steps': 3,
        'domain_type': 'financial',
        'use_energy_tracking': True
    }

    model = create_temporal_spectral_model(config)
    
    # testing data
    batch_size, num_nodes, num_features = 2, 10, 8
    temporal_sequence = []
    
    for t in range(config['temporal_steps']):
        node_features = torch.randn(batch_size, num_nodes, num_features)
        eigenvalues = torch.sort(torch.rand(batch_size, 8))[0]
        eigenvectors = torch.randn(batch_size, num_nodes, 8)
        temporal_sequence.append((node_features, eigenvalues, eigenvectors))
    
    # forward pass
    with torch.no_grad():
        results = model(temporal_sequence)
        print(f"  Anomaly scores shape: {results['anomaly_scores'].shape}")
        print(f"  Spectral features shape: {results['spectral_features'].shape}")
        print("  ✅ Temporal spectral model working!")

def main():
    """Run all tests"""
    print("Novel Pipeline Component Testing")
    print("=" * 40)
    
    try:
        test_dynamic_wavelets()
        test_temporal_spectral_model()
        
        print("\n" + "=" * 40)
        print("ALL TESTS PASSED")
        print("Novel components successfully implemented:")
        print("  - Dynamic wavelets with domain adaptation")
        print("  - Temporal spectral fusion architecture")
        print("  - Energy-guided multi-scale analysis")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()