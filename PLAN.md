# BlackBox AutoML — Project Plan

## 🎯 Goal
Build an automated ML pipeline where a user drops a CSV file into an `input/` folder, runs a single command (Docker or Python), and receives a fully documented Jupyter Notebook explaining every step: **data cleansing → data splitting → training → prediction → recommendations**.

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
│   ├── training.py           # Model training logic
│   ├── prediction.py         # Prediction & evaluation
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

### 4. **Training** → `src/training.py`
- **Classification**: Logistic Regression, Random Forest, XGBoost, SVM
- **Regression**: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- Hyperparameter tuning via GridSearchCV (optional)
- Cross-validation (k-fold)

### 5. **Prediction & Evaluation** → `src/prediction.py`
- Generate predictions on test set
- Classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
- Regression metrics: MAE, MSE, RMSE, R²
- Feature importance extraction

### 6. **Recommendations** → `src/recommendation.py`
- Select best performing model
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
   OR: python src/pipeline.py
4. Open output/automl_report.ipynb in Jupyter/VSCode
```

---

## 📦 Dependencies
- pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, nbformat, jupyter

---

## ✅ Implementation Status

- [x] Project structure and .gitignore
- [x] src/data_cleaning.py — data loading, missing values, duplicates, outliers
- [x] src/data_split.py — feature/target split, train/test split, problem type detection
- [x] src/training.py — multi-model training, cross-validation, hyperparameter tuning
- [x] src/prediction.py — evaluation metrics, feature importance, best model selection
- [x] src/recommendation.py — natural language recommendations and insights
- [x] src/notebook_generator.py — full .ipynb generation with embedded plots
- [x] src/pipeline.py — main orchestrator tying everything together
- [x] templates/report_template.ipynb.json — notebook skeleton
- [x] requirements.txt — all Python dependencies
- [x] Dockerfile — container definition
- [x] docker-compose.yml — one-command Docker setup
- [x] README.md — comprehensive usage documentation
- [x] PLAN.md — this planning document
- [x] Git repository initialized