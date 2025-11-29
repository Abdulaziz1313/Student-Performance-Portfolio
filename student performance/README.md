# Student Performance Predictor 🎓  
*A Machine Learning Approach to Identifying At-Risk Students*

## 1. Project Overview

This project is part of my **Data Science & Machine Learning portfolio**.  
The goal is to build a **student performance predictor** that classifies whether a student is likely to **pass or fail** based on demographic, social, and academic features.

Using the **UCI Student Performance dataset**, I:

- Performed **exploratory data analysis (EDA)** to understand patterns in the data.
- Preprocessed and encoded features for machine learning.
- Trained and compared multiple **classification models**.
- Selected a **Random Forest** model as the final classifier.
- Used **SHAP** and **feature importance** to interpret the model’s decisions.
- Saved the trained model for reuse as `best_model.pkl`.

This repository demonstrates my skills in:

- Data handling & infrastructure  
- Data storage & access  
- Programming in Python  
- Machine learning model development  
- Model evaluation & explainability  

---

## 2. Professional Details

**Name:** Abdulaziz Aloufi  
**Student ID:** C00266252  
**Email:** mr.alofi19@gmail.com  
**GitHub:** https://github.com/Abdulaziz1313  

---

## 3. Dataset

- **Source:** UCI Machine Learning Repository – Student Performance Dataset  
- **File used:** `student-mat.csv` (mathematics performance)  
- **Size:** 395 students, 33 original features  
- **Target label:**  
  - Created a binary column `pass_fail` from final grade `G3`  
  - `pass_fail = 1` if `G3 >= 10` (pass)  
  - `pass_fail = 0` otherwise (fail)

### Main Feature Groups

- **Demographic & social**:  
  `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, etc.

- **Study-related**:  
  `studytime`, `failures`, `absences`, `higher`, `schoolsup`, etc.

- **Grades**:  
  `G1`, `G2`, `G3` (used only to create the label, then dropped to avoid leakage)

---

## 4. Project Structure (Infrastructure for Data)

```text
student-performance-portfolio/
│
├── .venv/                      # local virtual environment (not committed to Git)
│
├── data/
│   └── student-mat.csv         # raw dataset
│
├── notebooks/
│   ├── EDA.ipynb               # exploratory data analysis
│   ├── model_training.ipynb    # model development & evaluation
│   └── explainability.ipynb    # SHAP & feature importance
│
├── models/
│   └── best_model.pkl          # saved Random Forest model
│
├── reports/
│   ├── single_item_report.pdf  # first 10% portfolio item
│   └── final_report.pdf        # full portfolio report
│
├── requirements.txt            # Python dependencies
└── README.md                   # this file

