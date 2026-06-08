# BlackBox AutoML 🤖

**An automated machine learning pipeline powered by [AutoGluon](https://auto.gluon.ai/).**

Drop a CSV file into the `input/` folder, run a single command, and receive a fully documented Jupyter Notebook explaining every step of the ML pipeline — from data cleaning through model training, evaluation, and recommendations.

---

## 🚀 Quick Start

### With Python (recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your CSV to the input folder
cp your_data.csv input/

# 3. Run the pipeline (lightweight mode — fast)
python src/pipeline.py

# 4. Open the generated notebook
jupyter notebook output/automl_report.ipynb
```

### With Docker

```bash
# 1. Copy your CSV to the input folder
cp your_data.csv input/

# 2. Build and run
docker-compose up --build
```

The generated notebook will appear in `output/automl_report.ipynb`.

---

## 🧠 How It Works

BlackBox AutoML uses **AutoGluon** as its core ML engine. AutoGluon is a state-of-the-art automated machine learning framework that:

1. **Automatically detects** whether your problem is classification or regression
2. **Trains multiple models** with hyperparameter tuning built-in
3. **Builds ensembles** (weighted averaging, stacking) for superior performance
4. **Returns a leaderboard** ranking all trained models by performance
5. **Provides feature importance** analysis (permutation-based)

### The 6-Step Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────────┐
│  1.Load  │ →  │  2.Clean │ →  │  3.Split │ →  │  4.Train    │ →  │  5.Eval  │ →  │  6.Report    │
│  CSV     │    │  Data    │    │  Data    │    │  (AutoGluon)│    │  Models  │    │  + Recommend │
└──────────┘    └──────────┘    └──────────┘    └─────────────┘    └──────────┘    └──────────────┘
```

| Step | What Happens | Why It Matters |
|------|-------------|----------------|
| 1️⃣ | **Load CSV** — reads your file, checks headers, validates structure | Ensures your data is readable and correctly formatted |
| 2️⃣ | **Clean Data** — fills missing values, removes duplicates, caps outliers | Garbage in = garbage out. Clean data = better predictions |
| 3️⃣ | **Split Data** — auto-detects target column & problem type, splits 80/20 | Reserves unseen data for honest evaluation |
| 4️⃣ | **Train Models** — AutoGluon trains multiple models, tunes hyperparameters, builds ensembles | Finds the best predictor without manual trial-and-error |
| 5️⃣ | **Evaluate** — computes metrics, confusion matrix, residual plots, feature importance | Shows you exactly how each model performs |
| 6️⃣ | **Generate Notebook** — produces a self-contained .ipynb with all results | You get a complete, shareable report — no coding needed |

---

## 📐 Lightweight-First Progressive Model Strategy

The pipeline **starts with fast, low-resource models** and only escalates to more sophisticated ones when you ask:

| Quality Preset | CLI Flag | Models Trained | Expected Time | Best For |
|---------------|----------|----------------|---------------|----------|
| 🏃 **Lightweight** (default) | *(none)* | LinearModel, LightGBM, XGBoost, Random Forest | ~1–5 min | Quick baselines, large datasets, limited compute |
| ⚖️ **Balanced** | `--quality balanced` | Adds Extra Trees, KNN, medium ensembles | ~5–15 min | Better accuracy, moderate datasets |
| 🏆 **Best Quality** | `--quality best` | Full AutoGluon: Neural Nets, stacking, deep ensembles | ~15–60+ min | Maximum accuracy, smaller datasets, powerful hardware |

**💡 Pro tip:** Start with `lightweight` to validate your pipeline works, then escalate to `balanced` or `best` for final results.

---

## 🎯 CLI Usage

```
usage: python src/pipeline.py [-h] [--input INPUT] [--output OUTPUT]
                              [--target TARGET] [--test-size TEST_SIZE]
                              [--quality {lightweight,balanced,best}]
                              [--time-limit TIME_LIMIT]
                              [--missing-strategy {auto,mean,median,mode,drop}]
                              [--outlier-method {cap,remove,none}]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | *(scans input/)* | Path to your CSV file |
| `--output` | `-o` | `output/automl_report.ipynb` | Where to save the generated notebook |
| `--target` | `-t` | *(auto-detected)* | Name of the column you want to predict |
| `--test-size` | | `0.2` | Fraction of data reserved for testing (e.g., 0.2 = 20%) |
| `--quality` | `-q` | `lightweight` | Model quality: `lightweight` → `balanced` → `best` |
| `--time-limit` | | *(no limit)* | Max training time in seconds (e.g., 600 = 10 min) |
| `--missing-strategy` | | `auto` | How to handle missing values: auto, mean, median, mode, or drop |
| `--outlier-method` | | `cap` | How to handle outliers: cap, remove, or none |

### Real Examples

```bash
# 🔸 Fast baseline (use this first!)
python src/pipeline.py

# 🔸 Specify a file and target column explicitly
python src/pipeline.py -i input/my_data.csv -t price

# 🔸 Balanced quality with a 10-minute time limit
python src/pipeline.py -q balanced --time-limit 600

# 🔸 Full power — best accuracy (slowest)
python src/pipeline.py -q best

# 🔸 Custom output location
python src/pipeline.py -i input/data.csv -o ~/Desktop/my_report.ipynb
```

---

## 📁 Project Structure

```
blackbox-automl/
├── input/                    # ← Drop your CSV files here
├── output/                   # ← Generated notebook appears here
├── src/
│   ├── pipeline.py           # Main orchestrator — ties everything together
│   ├── data_cleaning.py      # Handles missing values, duplicates, outliers
│   ├── data_split.py         # Detects target, splits train/test
│   ├── training.py           # AutoGluon model training 🧠
│   ├── prediction.py         # AutoGluon evaluation & leaderboard
│   ├── recommendation.py     # Generates natural language insights
│   └── notebook_generator.py # Produces the .ipynb report
├── docker-compose.yml        # One-command Docker setup
├── Dockerfile                # Container definition
└── requirements.txt          # Python dependencies
```

---

## 📦 Dependencies

- **[autogluon.tabular](https://auto.gluon.ai/)** — Core AutoML engine (includes pandas, numpy, scikit-learn, xgboost, lightgbm)
- **matplotlib** + **seaborn** — Charts and visualizations in the report
- **nbformat** + **jupyter** — Notebook generation and viewing

---

## 📊 Example Output (Telco Customer Churn)

After running the pipeline on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, the generated notebook contains:

### Cleaning Summary
```
✅ Removed 0 duplicate rows
✅ Capped 1142 outliers in SeniorCitizen
✅ Dataset: 7043 rows × 21 columns
```

### AutoGluon Leaderboard
| Model | Score |
|-------|-------|
| 🥇 WeightedEnsemble_L2 | **0.7970** |
| 🥈 XGBoost | 0.7928 |
| 🥉 LightGBM | 0.7921 |
| LinearModel | 0.7921 |
| RandomForest | 0.7771 |

### Evaluation Metrics (Best Model)
- **Accuracy**: 79.70%
- **Precision**: 0.797 (weighted)
- **Recall**: 0.797 (weighted)
- **F1 Score**: 0.797 (weighted)

### Feature Importance
Top predictors of customer churn: Contract type, tenure, MonthlyCharges, InternetService, PaymentMethod

### Recommendation Verdict
> 👍 **Good.** The model achieves accuracy of 0.797. It's a solid baseline but improvements can be made. Try `--quality best` for better results.

---

## 💡 Tips for Best Results

| Tip | Why |
|-----|-----|
| **Clean your data first** | Remove obvious outliers, fix label errors — AutoGluon works best with clean input |
| **Use meaningful column names** | AutoGluon handles categorical features better when names are human-readable |
| **Start lightweight** | Validate that your pipeline runs before spending compute on `--quality best` |
| **Set a time limit** | `--time-limit 600` prevents runaway training on large datasets |
| **Name your target column clearly** | Common names like `target`, `label`, `price`, `churn` are auto-detected |
| **Check the notebook** | The generated .ipynb explains every step in detail — you don't need to be an ML expert |

---

## ⚠️ Disclaimer & Limitations

**Important information about using BlackBox AutoML:**

1. **Auto-generated reports**: The generated notebook is produced algorithmically. While it provides a thorough analysis, it is **not a substitute for professional data science review**. Always validate results before making business decisions.

2. **Model performance is not guaranteed**: AutoGluon uses automated machine learning techniques that work well on many datasets, but results vary depending on data quality, problem complexity, and available compute resources. A score of 0.80 on one dataset may be excellent or mediocre depending on the domain.

3. **Data privacy**: All processing happens **locally on your machine**. Your data is never sent to external servers. The CSV file you provide stays entirely within your environment.

4. **Resource usage**: AutoGluon can be compute-intensive, especially with `--quality best`. On machines with limited RAM (< 8 GB), stick with `lightweight` mode to avoid out-of-memory errors.

5. **Disk space**: AutoGluon saves model files temporarily during training. Ensure you have at least 1-2 GB of free disk space. Model files are automatically cleaned up after the pipeline completes.

6. **No warranty**: This tool is provided "as is" without warranty of any kind, either express or implied. The authors are not responsible for any damages or losses arising from its use.

7. **Third-party dependencies**: This project relies on AutoGluon, which in turn depends on various open-source libraries (scikit-learn, xgboost, lightgbm, etc.). Each has its own license and terms of use.

8. **Not for production**: This pipeline is designed for **exploratory analysis, learning, and prototyping**. For production deployments, additional considerations (model monitoring, data drift detection, A/B testing, compliance) are needed.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. When contributing:

- Follow the existing code style (PEP 8)
- Add docstrings to new functions
- Test your changes with a sample dataset

---

## 📄 License

MIT License — see LICENSE file for details.