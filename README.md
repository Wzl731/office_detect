# Office Malware Detection System

A machine learning-based system for detecting malicious macros in Microsoft Office documents.

## Features

- **Multiple ML Models**: Uses ensemble of RandomForest, MLP, KNN, and SVM classifiers
- **VBA Feature Extraction**: Extracts comprehensive features from VBA macro code
- **Ensemble Voting**: Combines predictions from multiple models for improved accuracy
- **Batch Processing**: Process entire folders of Office documents
- **Command Line Interface**: Easy-to-use CLI for batch analysis

## Requirements

- Python 3.7+
- Required packages:
  ```
  numpy
  pandas
  scikit-learn
  pickle
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Wzl731/office_detect.git
   cd office_detect
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```

## Usage

### Basic Usage

Analyze a folder of Office documents:

```bash
python detector.py --folder /path/to/documents
```

### Command Line Options

- `--folder, -f`: Path to folder containing Office documents to analyze
- `--models-dir, -m`: Path to trained models directory (default: models_0711)
- `--no-save`: Disable saving of classified files
- `--save-type`: Choose what to save: `all`, `malicious`, or `benign` (default: all)

### Examples

```bash
# Analyze documents in 'samples' folder
python detector.py --folder samples

# Use custom models directory
python detector.py --folder samples --models-dir my_models

# Only save malicious files
python detector.py --folder samples --save-type malicious

# Analyze without saving any files
python detector.py --folder samples --no-save
```

## How It Works

1. **Feature Extraction**: The system extracts various features from VBA macro code including:
   - API calls and function usage
   - String patterns and keywords
   - Code structure metrics
   - Suspicious behavior indicators

2. **Model Ensemble**: Four different machine learning models analyze the features:
   - Random Forest
   - Multi-Layer Perceptron (MLP)
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)

3. **Voting Decision**: The final classification is based on majority voting from all models

## Output

The system provides:
- Individual model predictions with confidence scores
- Ensemble voting results
- Classification statistics
- Automatic file organization (malicious files saved to `data/good2bad2`, benign files to `data/bad2good2`)

## Model Training

The system uses pre-trained models located in the `models_0711` directory. Models were trained on a comprehensive dataset of benign and malicious Office documents.

## File Structure

```
office_detect/
├── detector.py              # Main detection script
├── original_feature_extractor.py  # Feature extraction module
├── feature222.py           # VBA feature extractor
├── models_0711/            # Pre-trained models directory
├── data/                   # Output directory for classified files
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License

This project is for educational and research purposes.

## Disclaimer

This tool is designed for legitimate security research and malware analysis purposes. Users are responsible for ensuring compliance with applicable laws and regulations.
