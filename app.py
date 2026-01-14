import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="AI Career Coach", page_icon="🎓", layout="centered")

# --- 2. DEFINE OPTIMIZER CLASS (Internal Logic) ---
# We paste the class here so we don't need separate files, making deployment safer.
class CareerOptimizer:
    def __init__(self, model, feature_names, actionable_features, constraints):
        self.model = model
        self.feature_names = feature_names
        self.actionable = actionable_features
        self.constraints = constraints
        self.population_size = 50
        self.generations = 30

    def _get_fitness(self, individual, original):
        prob = self.model.predict_proba(individual.reshape(1, -1))[0][1]
        
        # LOGIC FIX: Penalize increasing backlogs
        if 'backlogs' in self.actionable:
            b_idx = self.feature_names.index('backlogs')
            if individual[b_idx] > original[b_idx]:
                 return -1000, prob

        if prob < 0.5:
            return prob * 1000, prob 
            
        effort = 0
        for feat in self.actionable:
            idx = self.feature_names.index(feat)
            range_val = self.constraints[feat][1] - self.constraints[feat][0]
            change = abs(individual[idx] - original[idx])
            effort += (change / range_val) * 10 
            
        return (prob * 1000) - effort, prob

    def optimize(self, student_vector):
        population = []
        for _ in range(self.population_size):
            ind = student_vector.copy()
            feat = random.choice(self.actionable)
            idx = self.feature_names.index(feat)
            low, high = self.constraints[feat]
            if feat == 'backlogs':
                if ind[idx] > 0: ind[idx] = random.randint(0, int(ind[idx]))
            else:
                ind[idx] = random.uniform(low, high)
            population.append(ind)

        best_solution = None
        best_prob = 0.0

        for gen in range(self.generations):
            scored_pop = []
            for ind in population:
                fit, prob = self._get_fitness(ind, student_vector)
                scored_pop.append((fit, ind, prob))
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            if scored_pop[0][2] > best_prob:
                best_prob = scored_pop[0][2]
                best_solution = scored_pop[0][1]
            
            survivors = [x[1] for x in scored_pop[:15]]
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                parent = random.choice(survivors)
                child = parent.copy()
                if random.random() < 0.4: 
                     feat = random.choice(self.actionable)
                     idx = self.feature_names.index(feat)
                     low, high = self.constraints[feat]
                     if feat == 'backlogs':
                         curr = int(child[idx])
                         if curr > 0: child[idx] = random.randint(0, curr)
                     else:
                         child[idx] = np.clip(child[idx] + random.randint(-5, 5), low, high)
                new_pop.append(child)
            population = new_pop
            
        return best_solution, best_prob

# --- 3. LOAD MODEL (Cached for Speed) ---
@st.cache_resource
def load_brain():
    try:
        model = joblib.load('model.pkl')
        columns = joblib.load('columns.pkl')
        return model, columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, model_columns = load_brain()

# --- 4. UI LAYOUT ---
st.title("🎓 AI Career Placement Coach")
st.write("Enter your academic details to get a personalized placement strategy.")

with st.form("student_input"):
    col1, col2 = st.columns(2)
    with col1:
        ssc = st.number_input("SSC % (10th)", 40, 100, 60)
        hsc = st.number_input("HSC % (12th)", 40, 100, 60)
        degree = st.number_input("Degree %", 40, 100, 60)
        cgpa = st.number_input("Current CGPA", 0.0, 10.0, 7.0)
    with col2:
        tech_score = st.slider("Technical Skill Score", 0, 100, 50)
        soft_score = st.slider("Soft Skill Score", 0, 100, 50)
        internships = st.slider("Internships Completed", 0, 5, 0)
        backlogs = st.number_input("Current Backlogs", 0, 10, 0)
    
    submitted = st.form_submit_button("Analyze My Profile")

# --- 5. EXECUTION LOGIC ---
if submitted and model is not None:
    # Prepare Data
    input_data = {
        "ssc_percentage": ssc, "hsc_percentage": hsc, "degree_percentage": degree,
        "cgpa": cgpa, "technical_skill_score": tech_score, "soft_skill_score": soft_score,
        "internship_count": internships, "backlogs": backlogs,
        # Defaults
        "live_projects": 0, "work_experience_months": 0, "certifications": 0,
        "attendance_percentage": 75, "entrance_exam_score": 50, "gender": 1,
        "extracurricular_activities": 0
    }
    
    # Align Columns
    df = pd.DataFrame([input_data])
    try:
        df = df[model_columns]
        student_vector = df.values[0]
        
        # Optimize
        constraints = {
            'technical_skill_score': (0, 100), 'soft_skill_score': (0, 100),
            'internship_count': (0, 5), 'backlogs': (0, 5), 'cgpa': (0, 10)
        }
        actionable = list(constraints.keys())
        
        with st.spinner("AI is simulating your future..."):
            opt = CareerOptimizer(model, model_columns, actionable, constraints)
            improved_vector, new_prob = opt.optimize(student_vector)
        
        # Display Results
        st.divider()
        prob_percent = new_prob * 100
        
        if prob_percent > 80:
            st.success(f"✅ High Chance of Placement: {prob_percent:.1f}%")
        elif prob_percent > 50:
            st.warning(f"⚠️ Moderate Chance: {prob_percent:.1f}%")
        else:
            st.error(f"❌ Low Chance: {prob_percent:.1f}%")
            
        st.subheader("🚀 Your Action Plan")
        
        changes_found = False
        for i, col in enumerate(model_columns):
            if col in actionable:
                original = student_vector[i]
                improved = improved_vector[i]
                if abs(improved - original) > 0.5:
                    changes_found = True
                    # Formatting text
                    if col == 'backlogs':
                        msg = f"Clear {int(original - improved)} Backlogs"
                    else:
                        msg = f"Increase by {round(improved - original, 1)}"
                    
                    with st.expander(f"Improve {col.replace('_', ' ').title()}"):
                        st.write(f"**Current:** {original} → **Target:** {round(improved, 1)}")
                        st.caption(f"👉 {msg}")

        if not changes_found:
             st.info("Your profile is solid! Focus on maintaining these stats.")

    except Exception as e:
        st.error(f"An error occurred: {e}")