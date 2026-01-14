import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Data
try:
    df = pd.read_csv('student_academic_placement_performance_dataset.csv')
except FileNotFoundError:
    print("Error: csv file not found. Please make sure the dataset is in the same folder.")
    exit()

# 2. Preprocessing
df = df.drop(columns=['student_id', 'salary_package_lpa'])

# Encoders
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

le_extra = LabelEncoder()
df['extracurricular_activities'] = le_extra.fit_transform(df['extracurricular_activities'])

# Define X and y
X = df.drop(columns=['placement_status'])
y = df['placement_status']

# 3. Train Model
print("Training Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# 4. Save Model
joblib.dump(rf_model, 'model.pkl')
print("✅ Model saved as 'model.pkl'")

# 5. Save Column Order (Crucial for API)
joblib.dump(list(X.columns), 'columns.pkl')
print("✅ Column names saved as 'columns.pkl'")