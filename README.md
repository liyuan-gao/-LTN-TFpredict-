# Logic Tensor Networks for Transcription Factor Prediction (LTN-TFpredict)

## 📋 Overview

This repository implements **Logic Tensor Networks (LTN)** for transcription factor (TF) prediction in protein sequences. The project demonstrates the superiority of LTN's interpretable logical reasoning over traditional neural networks and other baseline approaches through comprehensive experimental comparisons.

### 🎯 Key Features

- **LTN Integration**: Combines neural networks with logical constraints for interpretable protein classification
- **Multiple Architectures**: CNN, BiLSTM, and CNN+BiLSTM backbones with LTN integration
- **Comprehensive Baselines**: Multi-loss and data augmentation baselines for fair comparison
- **Statistical Validation**: Paired t-tests and Wilcoxon signed-rank tests for significance testing
- **Motif-Based Analysis**: Focus on five key transcription factor motifs (ZF, LZ, BHLH, FH, WHH)

## 🏗️ Project Structure

```
LTN-TFpredict/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── 📁 Core Training Scripts
├── train_LTN.py                      # Main LTN training (CNN, BiLSTM, CNN+BiLSTM)
├── train_LTN_t-test.py               # LTN training with statistical tests
├── train_seq_baselines.py            # Non-LTN baselines (BiLSTM, CNN+BiLSTM)
├── train_multi-loss.py               # Multi-loss baseline without LTN
├── train_augmentation.py             # Motif-based data augmentation baseline
│
├── 📁 Data Loading & Utilities
├── fast_dataset_loader.py            # Optimized dataset loader
├── utils_data.py                     # Data filtering utilities
├── models_seq.py                     # BiLSTM and CNN+BiLSTM backbone architectures
│
├── 📁 Comparison & Analysis
├── compare_seq_ltn_vs_baseline.py    # Orchestrate LTN vs baseline comparisons
├── compare_methods.py                # Performance comparison utilities
│
├── 📁 Data
├── data/                             # Dataset directory (see Dataset section)
│   ├── TF/                           # Transcription factor protein sequences
│   ├── NTF/                          # Non-transcription factor sequences
│   └── 0620/                         # Additional dataset
│
└── 📁 Output
    ├── model/                        # Saved trained models
    ├── *.pkl                         # Training metrics and results
    └── *.pth                         # PyTorch model weights
```

## 🔧 Environment Requirements

### Python Version
- Python 3.8+ (recommended: 3.9 or 3.10)

### Core Dependencies
```bash
pip install torch torchvision torchaudio
pip install logic-tensor-networks  # or ltn
pip install biopython
pip install scikit-learn
pip install pandas numpy matplotlib
pip install scipy
```

### Complete Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/LTN-TFpredict.git
cd LTN-TFpredict

# Install dependencies
pip install -r requirements.txt

# For CUDA support (optional, for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Hardware Recommendations
- **RAM**: 16GB+ (for large protein datasets)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 10GB+ free space for datasets and models

## 📊 Dataset

### Data Structure
The project uses protein sequence datasets organized as follows:

```
data/
├── TF/           # Transcription Factor proteins (positive samples)
│   ├── *.fasta   # Individual protein sequence files
├── NTF/          # Non-Transcription Factor proteins (negative samples)
│   ├── *.fasta   # Individual protein sequence files
└── 0620/         # Additional validation dataset
    ├── *.fasta   # Mixed TF/NTF sequences
```

### Dataset Features
- **Sequence Length**: Fixed length (1280 amino acids)
- **Motif Labels**: Five key transcription factor motifs
  - **ZF**: Zinc Finger domains
  - **LZ**: Leucine Zipper domains
  - **BHLH**: Basic Helix-Loop-Helix domains
  - **FH**: Forkhead domains
  - **WHH**: Winged Helix-Turn-Helix domains
- **Format**: FASTA format with protein IDs and sequences

### Sample Statistics
- **Total Samples**: ~58,000 protein sequences
- **TF Samples**: ~6,800 transcription factors
- **NTF Samples**: ~51,000 non-transcription factors
- **Motif Distribution**: Varies by motif type (see dataset loading output)

## 🚀 Quick Start

### 1. Basic LTN Training
```bash
# Train LTN model with CNN backbone
python train_LTN.py

# Train LTN model with BiLSTM backbone
BACKBONE=bilstm python train_LTN.py

# Train LTN model with CNN+BiLSTM hybrid
BACKBONE=cnn_bilstm python train_LTN.py
```

### 2. Baseline Comparisons
```bash
# Train non-LTN baselines
python train_seq_baselines.py

# Train multi-loss baseline
python train_multi-loss.py

# Train data augmentation baseline
python train_augmentation.py
```

### 3. Statistical Analysis
```bash
# Run LTN training with statistical significance tests
python train_LTN_t-test.py
```

### 4. Comprehensive Comparison
```bash
# Compare all methods automatically
python compare_seq_ltn_vs_baseline.py
```

## 📈 Key Results

### Performance Comparison
| Model Type | Test Accuracy | Test F1 | Logical Satisfaction | Interpretability |
|------------|---------------|---------|---------------------|------------------|
| **LTN CNN** | **96.8%** | **95.1%** | **0.883** | ✅ **High** |
| LTN BiLSTM | 93.2% | 87.9% | 0.795 | ✅ **High** |
| LTN CNN+BiLSTM | 91.7% | 84.8% | 0.767 | ✅ **High** |
| Baseline BiLSTM | 93.0% | 88.2% | ❌ N/A | ❌ Low |
| Baseline CNN+BiLSTM | 92.2% | 87.2% | ❌ N/A | ❌ Low |
| Multi-Loss | 96.3% | 93.5% | ⚠️ Pseudo-Sat | ⚠️ Medium |
| Data Augmentation | 94.9% | 91.7% | ❌ N/A | ❌ Low |

### Key Insights
1. **LTN provides interpretable logical reasoning** unavailable in other methods
2. **Multi-loss and data augmentation** achieve high performance but **lack interpretability**
3. **Statistical tests confirm** significant improvements during LTN training
4. **CNN backbone** currently performs best with LTN integration

## 🔬 Model Architectures

### 1. LTN Models (`train_LTN.py`)
- **CNN**: Convolutional layers + LTN logical constraints
- **BiLSTM**: Bidirectional LSTM + LTN integration
- **CNN+BiLSTM**: Hybrid architecture with LTN reasoning

### 2. Baseline Models (`train_seq_baselines.py`)
- Pure neural networks without logical constraints
- Same architectures as LTN models for fair comparison

### 3. Multi-Loss Model (`train_multi-loss.py`)
- Main classifier + auxiliary motif classifiers
- Multiple loss functions but **no logical satisfaction**

### 4. Data Augmentation Model (`train_augmentation.py`)
- Motif-based positive sample augmentation
- High performance but **no interpretable reasoning**

## 📊 Evaluation Metrics

### Standard ML Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for TF prediction
- **Recall**: Sensitivity for TF detection
- **F1-Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Accounts for class imbalance

### LTN-Specific Metrics
- **Logical Satisfaction**: Constraint satisfaction levels (0-1)
- **Per-Rule Satisfaction**: Individual motif rule compliance
- **Constraint Violation Analysis**: Interpretable failure modes

### Statistical Validation
- **Paired t-tests**: Parametric significance testing
- **Wilcoxon signed-rank tests**: Non-parametric validation
- **Trend Analysis**: Pearson correlation with training progress

## 💻 Code Usage Examples

### Training Custom LTN Model
```python
from train_LTN import TFModel_CNN, TFModel_BiLSTM
import ltn

# Initialize LTN model
model = TFModel_CNN(input_size=(1, 1280))
P = ltn.Predicate(model)

# Train with logical constraints
# (see train_LTN.py for complete implementation)
```

### Loading Trained Models
```python
import torch
from train_LTN import TFModel_CNN

# Load saved model
model = TFModel_CNN(input_size=(1, 1280))
model.load_state_dict(torch.load('model/TF-LTN_model_v5.pth'))
model.eval()

# Make predictions
predictions = model(input_sequences)
```

### Statistical Analysis
```python
from train_LTN_t_test import perform_statistical_tests

# Analyze training metrics
results = perform_statistical_tests(test_accuracies, "accuracy")
print(f"Improvement significance: p = {results['t_pvalue']:.6f}")
```

## 🎓 Research Applications

### Academic Use
- **Interpretable AI**: Demonstrates LTN's logical reasoning capabilities
- **Protein Bioinformatics**: TF prediction with biological constraints
- **Comparative Studies**: Baseline comparisons for fair evaluation

### Practical Applications
- **Drug Discovery**: Interpretable TF target identification
- **Regulatory Analysis**: Understanding transcriptional control
- **Biomarker Discovery**: Logical constraint-based feature selection

## 🤝 Contributing

### Issues and Feature Requests
- Report bugs via GitHub Issues
- Request features with clear use cases
- Provide datasets for testing new scenarios

### Code Contributions
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Coding Standards
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include unit tests for new features
- Update README for significant changes

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ltn_tfpredict2024,
    title={Logic Tensor Networks for Interpretable Transcription Factor Prediction},
    author={Your Name and Collaborators},
    journal={Journal Name},
    year={2024},
    volume={XX},
    pages={XXX-XXX}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks)
- [PyTorch](https://pytorch.org/)
- [BioPython](https://biopython.org/)
- [Scikit-learn](https://scikit-learn.org/)

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Email**: liygao@ttu.edu
- **Documentation**: Check code comments and docstrings

---

## 🏆 Acknowledgments

- Logic Tensor Networks research community
- Protein bioinformatics datasets
- PyTorch and scientific Python ecosystem
- Contributors and collaborators

**⭐ Star this repository if you find it useful!** 
