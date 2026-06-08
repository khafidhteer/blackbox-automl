# BlackBox AutoML — Project Plan

## 🎯 Goal
Build an automated ML pipeline where a user drops a CSV file into an `input/` folder, runs a single command (Docker or Python), and receives a fully documented Jupyter Notebook explaining every step: **data cleansing → data splitting → training → prediction → recommendations**.

The ML engine is **[AutoGluon](https://auto.gluon.ai/)** — an industry-standard AutoML framework that automatically handles model selection, hyperparameter tuning, ensembling, and evaluation with minimal code.

---

## 📁 Repository Structure

```
blackbox-automl/
├── input/                    # User drops CSV files here
│   └── .gitkeep
├── output/                   # Generated notebook goes here
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # Main orchestration script
│   ├── data_cleaning.py      # Data cleansing logic
│   ├── data_split.py         # Train/test split logic
│   ├── training.py           # AutoGluon model training 🧠
│   ├── prediction.py         # AutoGluon evaluation & leaderboard
│   ├── recommendation.py     # Model recommendation logic
│   └── notebook_generator.py # Generates the final .ipynb file
├── templates/
│   └── report_template.ipynb.json  # Base notebook template
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # One-command Docker run setup
├── README.md                 # Usage instructions
├── PLAN.md                   # This plan document
└── .gitignore
```

---

## 🧩 Component Breakdown

### 1. **Input Layer** (`input/`)
- User copies CSV files into this folder
- The script scans for the newest `.csv` file (or user specifies via CLI argument)
- Validates the CSV: checks for headers, non-empty rows

### 2. **Data Cleansing** → `src/data_cleaning.py`
- Load CSV file
- Handle missing values (median for numeric, mode for categorical)
- Remove duplicate rows
- Detect and cap outliers using IQR method
- Generate basic statistics

### 3. **Data Splitting** → `src/data_split.py`
- Auto-detect target column (last column or common names)
- Separate features (X) and target (y)
- Auto-detect problem type (classification vs regression) — maps to AutoGluon's `binary`, `multiclass`, or `regression`
- Drop high-cardinality columns (e.g., ID fields with >100 unique values) to prevent explosion from one-hot encoding
- Train/test split (80/20 default, stratified for classification)

### 4. **Training** → `src/training.py` (AutoGluon-powered)
- Uses **AutoGluon `TabularPredictor`** as the ML engine
- Automatically detects problem type — maps `classification` → `binary` (2 classes) or `multiclass` (3+), `regression` → `regression`
- **Lightweight-first progressive model strategy** with restricted model types:

| Mode | CLI Flag | Models Trained | Expected Time | Best For |
|------|----------|----------------|---------------|----------|
| **Lightweight** (default) | *(none)* | GBM, CAT, RF, LR, XGB — only fast, low-resource models | ~1-5 min | Quick baselines, large datasets, limited RAM |
| **Balanced** | `--quality balanced` | GBM, CAT, RF, XGB, XT, KNN, LR — medium complexity | ~5-15 min | Better accuracy, moderate datasets |
| **Best Quality** | `--quality best` | Full AutoGluon suite (no restrictions) | ~15-60+ min | Maximum accuracy, powerful hardware |

- Hyperparameter tuning is built into AutoGluon's model training
- No manual GridSearchCV or cross-validation loops needed
- Uses `autogluon.tabular` (not full `autogluon`) — smaller install, fewer dependencies

### 5. **Prediction & Evaluation** → `src/prediction.py`
- Uses AutoGluon's built-in evaluation:
  - `predictor.leaderboard()` — display all model results with metrics on test data
  - `predictor.predict()` — generate predictions per model
  - `predictor.feature_importance()` — permutation-based importance
- Also computes standard sklearn metrics for report:
  - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
  - Regression: MAE, MSE, RMSE, R², Residuals
- Best model automatically selected from leaderboard (or via AutoGluon's `model_best`)

### 6. **Recommendations** → `src/recommendation.py`
- Select best performing model from all_metrics dict
- Generate natural language insights with model-specific improvement suggestions:
  - Suggests `--quality balanced` or `--quality best` for low scores
  - Highlights top features and class imbalance
  - Warns about residual bias for regression
- Overall verdict (Outstanding/Great/Good/Fair/Poor)

### 7. **Notebook Generator** → `src/notebook_generator.py`
- Reads template `.ipynb` JSON
- Populates with **markdown explanations** + **executable Python code cells** + **embedded visualizations (base64 images)**
- Handles both AutoGluon-style training results (dict with `predictor`/`leaderboard`) and legacy format
- Each section has both explanatory text and executed results
- **🔑 Key Requirement — Fully Executable Notebook**: Unlike a static report, every code cell must contain **actual, runnable Python code** — not skeleton or placeholder code. The notebook must be fully executable end-to-end by the user in Jupyter/VSCode.
  - **Data Loading**: Code cell reads the actual CSV file from `input/` using `pd.read_csv()` with a user-configurable path
  - **Data Cleaning**: Code performs missing value handling (median/mode imputation), duplicate removal, and IQR-based outlier capping directly — not just displays pre-computed results
  - **EDA**: Code generates statistics via `df.describe()`, correlation heatmaps, and distribution plots using matplotlib/seaborn directly
  - **Data Splitting**: Code performs train/test split using `sklearn.model_selection.train_test_split()`, detects problem type automatically, and handles high-cardinality column exclusion
  - **Training**: Code instantiates and runs **AutoGluon `TabularPredictor`** with the specified quality preset and time limit (e.g., `TabularPredictor(label=target_col, problem_type=problem_type).fit(X_train, presets=presets)`)
  - **Evaluation**: Code calls `predictor.leaderboard()`, `predictor.evaluate()`, and `predictor.feature_importance()` directly against the test set
  - **Recommendations**: Code analyzes model performance from the leaderboard and generates insights programmatically
- **Output Pre-population (optional)**: Pre-computed outputs (text, images, dataframes) are embedded as cell _outputs_ so the notebook looks complete on first open, but the code is real and can be re-executed to regenerate results
- **Idempotency**: Running all cells from top to bottom must produce equivalent results without errors
- **CSV Path Handling**: The notebook accepts the CSV path as a variable at the top of the notebook (e.g., `CSV_PATH = "../input/dataset.csv"`) for easy user modification, with a relative path that works when the notebook is opened from the repo root
- **Runtime Mode**: Because AutoGluon training can take significant time when re-executed, the notebook includes a configurable `RUN_TRAINING = False` flag at the top — when `False`, cells load pre-computed results; when `True`, the notebook re-runs the full pipeline

---

## 🚀 User Workflow

```
1. Clone repo
2. Copy CSV file → input/ folder
3. Run: docker-compose up --build
   OR: python src/pipeline.py [--quality lightweight|balanced|best] [--target COLUMN] [--time-limit SECONDS]
4. Open output/automl_report.ipynb in Jupyter/VSCode
```

---

## 📦 Dependencies
- **autogluon.tabular** (bundles pandas, numpy, scikit-learn, xgboost, lightgbm)
- matplotlib, seaborn, nbformat, jupyter

---

## ⚙️ Key Technical Decisions & Fixes

| Issue | Solution |
|-------|----------|
| AutoGluon uses `binary`/`multiclass` not `classification` | Map `classification` → `binary` (2 classes) or `multiclass` (3+) in training.py |
| AutoGluon 1.5 model type keys changed | Use `GBM`, `CAT`, `RF`, `XGB`, `LR`, `XT`, `KNN` — not `LightGBM`, `SVM`, etc. |
| High-cardinality ID columns explode features | Drop columns with >100 unique values in data_split.py |
| xgboost version incompatibility | Require xgboost >=2.2 in requirements (handled by autogluon.tabular) |
| Disk space — full `autogluon` install is huge (~4 GB with PyTorch) | Use `autogluon.tabular` instead (~100 MB, no PyTorch needed for tabular) |
| Time-limit validation in AutoGluon | Set `--time-limit` to at least 180s for small datasets; training takes overhead time to initialize |

---

## ✅ Implementation Status

- [x] Project structure and .gitignore
- [x] src/data_cleaning.py — data loading, missing values, duplicates, outliers
- [x] src/data_split.py — feature/target split, train/test split, problem type detection, high-cardinality exclusion
- [x] src/training.py — **AutoGluon TabularPredictor with binary/multiclass mapping, lightweight-first presets**
- [x] src/prediction.py — **AutoGluon leaderboard, per-model evaluation, feature importance**
- [x] src/recommendation.py — **AutoGluon-aware recommendations with quality escalation suggestions**
- [x] src/notebook_generator.py — **✅ REWORK COMPLETE: all code cells now contain fully executable Python code**
- [x] src/pipeline.py — **✅ Updated to pass new params, save pipeline state for notebook re-execution**
- [x] templates/report_template.ipynb.json — notebook skeleton (no changes needed)
- [x] requirements.txt — **autogluon.tabular** (core ML engine) + visualization + notebook
- [x] Dockerfile — Slim Python 3.10 image with autogluon.tabular
- [x] docker-compose.yml — one-command Docker setup
- [x] README.md — **Full documentation with disclaimer, real Telco Churn results, tips**
- [x] PLAN.md — this planning document (updated with implementation details)
- [x] Git repository initialized

## 🧪 Verified With

- **Dataset**: Telco Customer Churn (7043 rows, 21 columns, binary classification)
- **Result**: 5 models trained in ~180s, best accuracy 0.797 (WeightedEnsemble_L2)
- **Output**: Full Jupyter notebook at `output/automl_report.ipynb` (309 KB)

---

## 📋 Action Items — `src/notebook_generator.py` Rework for Executable Notebooks

The following changes are needed to `src/notebook_generator.py` so that the generated `.ipynb` is a fully executable notebook rather than a static report.

### ✅ All Action Items Completed

All the changes described in the action items above have been implemented in:

| File | What Changed |
|------|-------------|
| `src/notebook_generator.py` | All section generators now output fully executable Python code instead of skeleton/placeholder code. Added config header, real cleaning code, live EDA computation, executable split/train/eval/recommendation cells. |
| `src/pipeline.py` | Now saves pipeline state (predictor + leaderboard) to `output/pipeline_state.pkl` for notebook re-execution. Passes all new parameters (`csv_path_for_nb`, `target_col`, `problem_type`, `quality_preset`, `time_limit`, `seed`, `test_size`) to `generate_notebook()`. Computes relative CSV path for notebook portability. |
| `templates/report_template.ipynb.json` | No changes needed — template structure was already correct. |

#### Notebook Executability Summary

| Section | Before | After |
|---------|--------|-------|
| **Config** | (missing) | First code cell with `CSV_PATH`, `TARGET_COL`, `PROBLEM_TYPE`, `QUALITY_PRESET`, `TIME_LIMIT`, `RUN_TRAINING`, `SEED`, `TEST_SIZE` |
| **Data Loading** | Hardcoded `info['shape']` strings | `pd.read_csv(CSV_PATH)` with live `df.shape`, `df.duplicated()`, `df.dtypes` |
| **Data Cleaning** | Pre-computed markdown only | Executable `fillna()`, `drop_duplicates()`, IQR clipping with `print()` output |
| **EDA** | `\\n` escaped newlines, hardcoded `numeric_cols` | Real `df.select_dtypes()`, proper newlines, live plot generation |
| **Data Splitting** | Markdown-only descriptions | Executable target detection, `train_test_split()`, auto problem type detection |
| **Training** | `predictor` not defined in notebook scope | Real `TabularPredictor(...).fit(...)` with `RUN_TRAINING` toggle + serialized state loading |
| **Evaluation** | `# Results summarized below` placeholders | Real `accuracy_score()`, `confusion_matrix()`, `predictor.feature_importance()` calls |
| **Recommendations** | `# Results displayed below` placeholders | Code reads from `leaderboard` DataFrame, evaluates thresholds, prints actionable suggestions |
