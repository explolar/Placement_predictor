from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from optimizer import CareerOptimizer

app = FastAPI()

try:
    model = joblib.load('model.pkl')
    model_columns = joblib.load('columns.pkl')
    print("✅ Model & Columns loaded.")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None

class StudentProfile(BaseModel):
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    technical_skill_score: float
    soft_skill_score: float
    internship_count: int
    backlogs: int
    live_projects: int = 0
    work_experience_months: int = 0
    certifications: int = 0
    attendance_percentage: float = 75.0
    entrance_exam_score: float = 50.0 
    gender: int = 1
    extracurricular_activities: int = 0

@app.post("/advise")
def get_career_advice(profile: StudentProfile):
    if not model:
        raise HTTPException(status_code=500, detail="Model missing.")

    df = pd.DataFrame([profile.dict()])
    try:
        df = df[model_columns]
    except KeyError as e:
        return {"error": f"Column mismatch: {e}"}

    student_vector = df.values[0]
    
    constraints = {
        'technical_skill_score': (0, 100), 'soft_skill_score': (0, 100),
        'internship_count': (0, 5), 'backlogs': (0, 5), 'cgpa': (0, 10)
    }
    actionable = list(constraints.keys())
    
    opt = CareerOptimizer(model, model_columns, actionable, constraints)
    improved_vector, new_prob = opt.optimize(student_vector)
    
    advice = []
    for i, col in enumerate(model_columns):
        if col in actionable:
            original = student_vector[i]
            improved = improved_vector[i]
            
            # Only show if change is significant
            if abs(improved - original) > 0.5:
                action_text = ""
                # TEXT FIX: Handle Backlogs separately
                if col == "backlogs":
                    diff = int(original - improved)
                    if diff > 0:
                        action_text = f"Clear {diff} Backlogs"
                    else:
                        continue # Don't show if backlogs increased (shouldn't happen now)
                else:
                    diff = round(improved - original, 1)
                    if diff > 0:
                        action_text = f"Increase by {diff}"
                    else:
                         continue # Don't advise decreasing skills

                advice.append({
                    "feature": col,
                    "current": float(round(original, 1)),
                    "target": float(round(improved, 1)),
                    "action": action_text
                })
                
    return {
        "current_status": "Not Placed" if new_prob < 0.5 else "Placed",
        "achievable_probability": f"{new_prob*100:.1f}%",
        "action_plan": advice
    }