# Explainable Graph Anomaly Detection (GAD) Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Graph Anomaly Detection pipeline with explainable AI capabilities using novel spectral learning methods and GraphSVX explanations.

## Key Features

- **Novel Spectral Methods**: Contrastive spectral learning and temporal spectral fusion
- **Explainable AI**: GraphSVX Shapley value explanations for predictions
- **Multi-Modal Detection**: Distance-based, spectral, and graph neural network approaches
- **Production Ready**: Complete evaluation, training, and deployment tools
- **Interactive Analysis**: Jupyter notebook for exploratory data analysis

## Quick Start

### Installation

```bash
git clone https://github.com/mweber54/explainable_gad.git
cd explainable_gad
pip install -r requirements.txt
```

### Basic Usage

```python
from models.models import *
from utils.evaluation import *
from utils.config import load_config

# Load dataset
data = torch.load('data/processed/weibo_static.pt', map_location='cpu')

# Run anomaly detection
config = load_config('data/configs/weibo_config.json')
model = create_anomaly_detector(config)
scores = model.predict(data)
```

### Exploratory Analysis

```bash
# Launch Jupyter notebook for interactive analysis
jupyter notebook notebook/exploratory_analysis.ipynb
```

## Repository Structure

```
explainable_gad/
├── data/                    # Dataset storage and configurations
│   ├── configs/            # Dataset configuration files
│   ├── raw/               # Raw datasets (gitignored)
│   └── processed/         # Processed datasets (gitignored)
├── models/                 # Model implementations
│   ├── models.py          # Core GAD models
│   ├── contrastive_spectral_learning.py
│   ├── temporal_spectral_fusion.py
│   ├── explainable_spectral_analysis.py
│   └── integrated_novel_pipeline.py
├── utils/                  # Utilities and helper functions
│   ├── config.py          # Configuration management
│   ├── evaluation.py      # Evaluation metrics
│   ├── graphsvx.py        # GraphSVX explainer
│   ├── model_wrapper.py   # Model interface
│   └── process_*.py       # Data preprocessing scripts
├── training/               # Training scripts and pipelines
│   └── training.py        # Main training pipeline
├── experiments/            # Experimental validation scripts
│   ├── main.py           # Main experiment runner
│   └── *.py              # Specific experiment scripts
├── tests/                  # Unit tests and validation
│   └── test_*.py         # Test files
├── notebook/               # Jupyter notebooks
│   └── exploratory_analysis.ipynb
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Performance Results

Our pipeline has been evaluated on 8 diverse graph datasets:

| Dataset | Domain | Nodes | ROC-AUC | Status |
|---------|--------|-------|---------|---------|
| **Weibo** | Social Media Bot Detection | 8,405 | **97.09%** | Excellent |
| **Amazon** | E-commerce Fraud | 9,744 | **94.80%** | Excellent |
| **T-Finance** | Financial Fraud | 39,357 | **88.87%** | Very Good |
| **Elliptic** | Bitcoin Analysis | 23,297 | **87.12%** | Very Good |
| **Yelp** | Review Spam | 45,954 | **74.32%** | Acceptable |
| **Questions** | Q&A Anomalies | 20,000 | **71.83%** | Acceptable |
| **Tolokers** | Crowdsourcing Fraud | 3,922 | **68.27%** | Needs Improvement |
| **Reddit** | Social Network | 5,468 | **65.21%** | Poor Performance |

## Novel Contributions

### 1. Contrastive Spectral Learning
- Custom spectral contrastive methods for graph anomaly detection
- Multi-scale frequency domain analysis

### 2. Temporal Spectral Fusion
- Temporal graph analysis with spectral methods
- Dynamic anomaly detection capabilities

### 3. Explainable Spectral Analysis
- Interpretable frequency domain methods
- Feature attribution in spectral space

### 4. Integrated Pipeline
- Unified framework combining all approaches
- Seamless model switching and comparison

## Explainable AI with GraphSVX

GraphSVX provides Shapley value explanations that answer: *"Why is this node anomalous?"*

### Example Usage

```python
from utils.graphsvx import GraphSVXExplainer
from utils.model_wrapper import create_model_wrapper

# Create model wrapper
model_wrapper = create_model_wrapper(
    model_type='distance_based',
    method_name='l2',
    train_data=X_train,
    train_labels=y_train
)

# Initialize explainer
explainer = GraphSVXExplainer(model=model_wrapper, data=data)

# Explain anomalous node
explanation = explainer.explain_node_comprehensive(node_idx=123)
print(f"Prediction: {explanation['original_prediction']:.4f}")
```

## Training and Evaluation

### Training Models

```python
from training.training import train_model
from utils.config import load_config

config = load_config('data/configs/weibo_config.json')
model = train_model(config)
```

### Running Experiments

```python
# Run comprehensive evaluation
python experiments/main.py --dataset weibo --method spectral_contrastive

# Run specific experiments
python experiments/run_dsgad_experiments.py
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_novel_pipeline.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@software{explainable_gad_2025,
  title={Explainable Graph Anomaly Detection with Novel Spectral Methods},
  author={Weber, M. and Contributors},
  year={2025},
  url={https://github.com/mweber54/explainable_gad},
  note={Graph Anomaly Detection Pipeline with Spectral Learning and Shapley Value Explanations}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Issues**: [GitHub Issues](https://github.com/mweber54/explainable_gad/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mweber54/explainable_gad/discussions)

---

**Star this repository if you find it useful!**

*Built for the graph anomaly detection and explainable AI community*