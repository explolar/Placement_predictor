import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. LOAD DATA
# Make sure your CSV file name is correct!
df = pd.read_csv('student_academic_placement_performance_dataset.csv')

# 2. CLEANING & PREPROCESSING
# Drop ID and Salary (we only want to predict Placement Status)
df = df.drop(columns=['student_id', 'salary_package_lpa'], errors='ignore')

# Encode 'gender' (Male/Female -> 1/0)
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

# Encode 'extracurricular_activities' (Yes/No -> 1/0)
le_extra = LabelEncoder()
df['extracurricular_activities'] = le_extra.fit_transform(df['extracurricular_activities'])

# Encode Target 'placement_status' (Placed/Not Placed -> 1/0)
le_target = LabelEncoder()
df['placement_status'] = le_target.fit_transform(df['placement_status'])

# 3. DEFINE FEATURES AND TARGET
X = df.drop(columns=['placement_status'])
y = df['placement_status']

# 4. TRAIN MODEL
# We use more trees (n_estimators=200) for better accuracy
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# 5. SAVE ARTIFACTS
# We save the model AND the column names so the App knows the exact order
joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'columns.pkl')

print("✅ Model Retrained Successfully!")
print(f"🧠 The Model now understands these {len(X.columns)} features:")
print(list(X.columns))