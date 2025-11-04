# =====================================================
# Student Performance Predictor
# Name: Abdulaziz Aloufi
# Student ID: C00266252
# Date: 4 November 2025
# =====================================================

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap

# --- 2. Load Dataset ---
df = pd.read_csv("data/student-mat.csv", sep=';')
  # path to my dataset
print("Shape:", df.shape)
df.head()
# Check missing values and data types
df.info()
df.isnull().sum().sum()
# Create pass/fail target
df['pass_fail'] = np.where(df['G3'] >= 10, 1, 0)  # 1 = pass, 0 = fail

# Drop unnecessary grade columns
df = df.drop(columns=['G1', 'G2', 'G3'])

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df.head()
# Split dataset
X = df.drop('pass_fail', axis=1)
y = df['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_log))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest Report ===")
print(classification_report(y_test, y_pred_rf))
# Confusion Matrix
def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, "Random Forest Confusion Matrix")
# Feature importance from Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top10.values, y=top10.index)
plt.title("Top 10 Important Features")
plt.show()

# SHAP explainability (optional, slower)
# explainer = shap.TreeExplainer(rf)
#shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values[1], X_test, show=False)
# plt.show()

