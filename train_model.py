import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import sys

# Force output to appear immediately
def log(message):
    print(message, flush=True)

if __name__ == "__main__":
    log("--- STARTED TRAINING SCRIPT ---")

    # 1. Load Data
    log("Loading data...")
    try:
        df = pd.read_csv('student_academic_placement_performance_dataset.csv')
    except FileNotFoundError:
        log("ERROR: 'student_academic_placement_performance_dataset.csv' not found.")
        input("Press Enter to exit...")
        sys.exit()

    # 2. Preprocessing
    df = df.drop(columns=['student_id', 'salary_package_lpa'])

    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])

    le_extra = LabelEncoder()
    df['extracurricular_activities'] = le_extra.fit_transform(df['extracurricular_activities'])

    X = df.drop(columns=['placement_status'])
    y = df['placement_status']

    # 3. Train
    log("Training Random Forest Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    # 4. Calculate Scores
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 5. Build Report Text
    output_text = []
    output_text.append("--- MODEL PERFORMANCE REPORT ---")
    output_text.append(f"Accuracy: {acc*100:.2f}%")
    output_text.append("\nConfusion Matrix:")
    output_text.append(f"True Negatives: {cm[0][0]}")
    output_text.append(f"False Positives: {cm[0][1]}")
    output_text.append(f"False Negatives: {cm[1][0]}")
    output_text.append(f"True Positives: {cm[1][1]}")
    output_text.append("\nDetailed Report:")
    output_text.append(report)

    final_report = "\n".join(output_text)

    # 6. Print to Console
    log(final_report)

    # 7. Save to File (Backup)
    with open("model_report.txt", "w") as f:
        f.write(final_report)
    log("\nReport saved to 'model_report.txt'")

    # 8. Save Model
    joblib.dump(rf_model, 'model.pkl')
    joblib.dump(list(X.columns), 'columns.pkl')
    log("Model saved successfully.")

    # 9. Wait for user
    input("\nScript finished. Press Enter to close this window...")