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
- Auto-detect problem type (classification vs regression)
- Train/test split (80/20 default, stratified for classification)

### 4. **Training** → `src/training.py` (AutoGluon-powered)
- Uses **AutoGluon `TabularPredictor`** as the ML engine
- Automatically detects problem type (classification / regression)
- **Lightweight-first progressive model strategy**:

| Mode | CLI Flag | Models Trained | Time / Resources |
|------|----------|----------------|------------------|
| **Lightweight** (default) | *(none)* | LinearModel, LightGBM (fast), CatBoost (small), XGBoost (small), RandomForest (small) | Fast (~1-5 min), low RAM |
| **Balanced** | `--quality balanced` | All default + medium ensembles, extra k-fold | Moderate (~5-15 min) |
| **Best Quality** | `--quality best` | Full suite: Neural Networks, stacking, deep ensembles, extra trees | High (~15-60+ min), high RAM |

- Hyperparameter tuning is built into AutoGluon's model training
- No manual GridSearchCV or cross-validation loops needed

### 5. **Prediction & Evaluation** → `src/prediction.py`
- Uses AutoGluon's built-in evaluation:
  - `predictor.leaderboard()` — display all model results with metrics
  - `predictor.evaluate()` — comprehensive scoring
  - `predictor.feature_importance()` — permutation-based importance
- Classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
- Regression metrics: MAE, MSE, RMSE, R²
- Best model automatically selected from leaderboard

### 6. **Recommendations** → `src/recommendation.py`
- Select best performing model from AutoGluon leaderboard
- Generate natural language insights
- Performance improvement suggestions
- Overall verdict (Outstanding/Great/Good/Fair/Poor)

### 7. **Notebook Generator** → `src/notebook_generator.py`
- Reads template `.ipynb` JSON
- Populates with markdown explanations + code cells + embedded visualizations
- Each section has both explanatory text and executed results

---

## 🚀 User Workflow

```
1. Clone repo
2. Copy CSV file → input/ folder
3. Run: docker-compose up --build
   OR: python src/pipeline.py [--quality balanced|best] [--target COLUMN]
4. Open output/automl_report.ipynb in Jupyter/VSCode
```

---

## 📦 Dependencies
- **autogluon** (bundles pandas, numpy, scikit-learn, xgboost, lightgbm, catboost)
- matplotlib, seaborn, nbformat, jupyter

---

## ✅ Implementation Status

- [x] Project structure and .gitignore
- [x] src/data_cleaning.py — data loading, missing values, duplicates, outliers
- [x] src/data_split.py — feature/target split, train/test split, problem type detection
- [ ] src/training.py — **Rewrite with AutoGluon TabularPredictor**
- [ ] src/prediction.py — **Rewrite with AutoGluon leaderboard/evaluate**
- [ ] src/recommendation.py — **Rewrite with AutoGluon metadata**
- [x] src/notebook_generator.py — full .ipynb generation with embedded plots
- [ ] src/pipeline.py — **Simplify orchestration with AutoGluon API**
- [x] templates/report_template.ipynb.json — notebook skeleton
- [ ] requirements.txt — **Replace scikit-learn + xgboost → autogluon**
- [ ] Dockerfile — **Update pip install for autogluon**
- [x] docker-compose.yml — one-command Docker setup
- [ ] README.md — **Reflect AutoGluon usage**
- [x] PLAN.md — this planning document
- [x] Git repository initialized