# ML Simple Trainer

Machine Learning training and prediction tool with CLI and Web UI.

## Features

- Train classification and regression models
- Support Random Forest and Logistic Regression
- CLI for developers and automation
- Web UI (Streamlit) for general users
- Auto-detect problem type (classification/regression)
- Save and load models with joblib

## Installation

```bash
# Clone repository
git clone https://github.com/giarHendris4/ML.git
cd ML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Command Line Interface (CLI)

Training:
```bash
python main.py --mode train --data data/raw/sample_classification.csv
```

Prediction:
```bash
python main.py --mode predict --data data/raw/new_data.csv
```

### Option 2: Web UI (Streamlit)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Configuration

Edit `config.yaml` to change default settings:

```yaml
target_column: null  # null = use last column
test_size: 0.2
random_state: 42
model_type: "random_forest"  # or "logistic_regression"
```

## Project Structure

```
ML/
├── app.py              # Streamlit web UI
├── main.py             # CLI entry point
├── config.yaml         # Configuration
├── requirements.txt    # Dependencies
├── src/                # ML modules
│   ├── data_loader.py
│   ├── trainer.py
│   └── predictor.py
├── data/raw/           # Data directory
├── models/             # Saved models
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Testing

Run all unit tests:
```bash
python -m unittest discover tests -v
```

## Sample Data

Sample files are provided in `data/raw/`:
- `sample_classification.csv` - 10 samples with binary labels
- `sample_regression.csv` - 15 samples with continuous target
- `new_data.csv` - 3 samples for prediction

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## License

MIT
