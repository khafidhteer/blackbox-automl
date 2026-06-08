# 🤖 BlackBox AutoML

**Automated Machine Learning Pipeline** — Drop a CSV file, run one command, get a fully documented Jupyter Notebook.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Place your CSV file in the input/ folder
cp your_data.csv input/

# 2. Run with Docker
docker-compose up --build

# 3. Open the generated notebook
jupyter notebook output/automl_report.ipynb
```

### Option 2: Python (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your CSV file in the input/ folder
cp your_data.csv input/

# 3. Run the pipeline
python src/pipeline.py

# 4. Open the generated notebook
jupyter notebook output/automl_report.ipynb
```

## 📋 Command Line Options

```bash
python src/pipeline.py [OPTIONS]

Options:
  -i, --input PATH         Path to input CSV file (auto-detects in input/ folder)
  -o, --output PATH        Output notebook path (default: output/automl_report.ipynb)
  -t, --target COLUMN      Target column name (auto-detected)
  --test-size FLOAT        Test set size ratio (default: 0.2)
  --tune                   Enable hyperparameter tuning (slower, better results)
  --no-scale               Disable feature scaling
  --missing-strategy STR   Missing value strategy: auto, mean, median, mode, drop
  --outlier-method STR     Outlier handling: cap, remove, none
```

### Examples

```bash
# Auto-detect everything
python src/pipeline.py

# Specify input file and target column
python src/pipeline.py -i input/my_data.csv -t price

# Enable hyperparameter tuning
python src/pipeline.py --tune

# Custom split and outlier handling
python src/pipeline.py --test-size 0.3 --outlier-method remove
```

## 📓 Generated Notebook

The output notebook (`output/automl_report.ipynb`) contains:

| Section | Description |
|---------|-------------|
| **1. Data Loading & Cleansing** | Load CSV, handle missing values, remove duplicates, detect/cap outliers |
| **2. Exploratory Data Analysis** | Statistics, correlation heatmap, distribution plots |
| **3. Data Splitting** | Feature/target separation, train/test split (80/20) |
| **4. Model Training** | Multiple models trained with cross-validation |
| **5. Model Evaluation** | Metrics, confusion matrix, residual analysis, feature importance |
| **6. Recommendations** | Best model selection, actionable insights, overall verdict |

Each section includes both **explanatory markdown** and **executed code cells** with results and visualizations.

## 🧠 Supported Models

### Classification
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- SVM (RBF Kernel)

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

## 📂 Project Structure

```
blackbox-automl/
├── input/                    # 📥 Place your CSV files here
├── output/                   # 📤 Generated notebook appears here
├── src/
│   ├── pipeline.py           # Main orchestrator
│   ├── data_cleaning.py      # Data cleansing logic
│   ├── data_split.py         # Train/test splitting
│   ├── training.py           # Model training
│   ├── prediction.py         # Evaluation & metrics
│   ├── recommendation.py     # Recommendations engine
│   └── notebook_generator.py # .ipynb generation
├── templates/
│   └── report_template.ipynb.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🐳 Docker Details

The Docker setup mounts two volumes:
- `./input` → `/app/input` (add your CSV files here)
- `./output` → `/app/output` (notebook appears here)

```bash
# Build and run
docker-compose up --build

# Or build first, then run with custom arguments
docker build -t blackbox-automl .
docker run --rm -v "%cd%/input:/app/input" -v "%cd%/output:/app/output" blackbox-automl python src/pipeline.py --tune
```

## 🛠 Requirements

- **Docker** (for Docker option) or **Python 3.9+**
- Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, nbformat, jupyter

## 📄 License

MIT