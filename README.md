![Image_Alt](https://github.com/AyushBhagwate/Wine-Classification-ML/blob/e192ad0bca0cec0a1a5c9c472eae3bd32e3669be/Wine-Classification-ML_Banner.png)

# 🍷 Wine Classification using Random Forest

## 📌 Overview
This project classifies different types of wine using a **Random Forest Classifier**. It demonstrates a complete machine learning workflow including **EDA, preprocessing, model training, hyperparameter tuning, and evaluation**.

---

## 🚀 Features
- 📊 Manual + Automated EDA
- ⚙️ Pipeline-based preprocessing
- 🌲 Random Forest Classifier
- 🔍 Hyperparameter tuning using GridSearchCV
- 📈 Model evaluation with multiple metrics
- 📊 Feature importance visualization using Seaborn

---

## 📂 Project Structure

wine_randomforest_pj/

├── data/

│   └── wine classification.csv

├── notebooks/

│   └── eda.ipynb

├── outputs/

│   ├── profile_report.html

│   ├── metrics.txt

│   └── predictions.csv

├── models/

│   └── best_model.pkl

├── src/

│   ├── __init__.py

│   ├── data_preprocessing.py

│   ├── train.py

│   ├── evaluate.py

│   ├── improve_model.py

├── creating_file.py

├── main.py

├── requirements.txt

└── README.md

---

## 🔍 Exploratory Data Analysis (EDA)

### 📘 Manual EDA
Located in: `notebooks/eda.ipynb`

Includes:
- Distribution analysis
- Correlation analysis
- Outlier detection
- Skewness analysis
- Feature relationships

### 🤖 Automated EDA
Generated using **ydata-profiling**

Saved at: `outputs/profile_report.html`

Provides:
- Feature summaries
- Missing value analysis
- Correlations
- Statistical insights
- Data quality checks

---

## 🌲 Model Used

### Random Forest Classifier
The project uses **Random Forest Classifier** for multiclass wine classification.

### Hyperparameter Tuning
Optimized using **GridSearchCV** with parameters such as:
- `n_estimators`
- `max_depth`
- `min_samples_split`

---

## 📈 Model Performance
- ✅ Accuracy: ~97% to 100%
- ✅ Strong precision & recall
- ✅ Robust multiclass classification performance

---

## 📊 Visualization

### 🔹 Feature Importance
- Built using Seaborn
- Displays the most influential wine features

---

## ▶️ How to Run

pip install -r requirements.txt

python main.py

---

## 🧠 Key Learnings
- Building ML pipelines
- Random Forest implementation
- Hyperparameter tuning with GridSearchCV
- Handling multiclass classification
- Combining manual + automated EDA
- Feature importance interpretation
- Model evaluation techniques

---

## 📌 Future Improvements
- Add confusion matrix visualization
- Try other ML models
- Improve model accuracy
- Deploy using Streamlit

---

## 👤 Author
**Ayush Bhagwate**
