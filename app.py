import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="AI Career Coach", page_icon="🎓", layout="centered")

# --- 2. DEFINE OPTIMIZER CLASS ---
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
        
        # Constraint: Penalize increasing backlogs
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
            
            # Logic: Backlogs only go down, integers stay integers
            if feat == 'backlogs':
                if ind[idx] > 0: ind[idx] = random.randint(0, int(ind[idx]))
            elif feat in ['internship_count', 'live_projects', 'certifications', 'work_experience_months', 'extracurricular_activities']:
                ind[idx] = random.randint(int(low), int(high))
            else:
                ind[idx] = random.uniform(low, high)
            population.append(ind)

        best_solution = student_vector.copy()
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
                     elif feat in ['internship_count', 'live_projects', 'certifications', 'work_experience_months', 'extracurricular_activities']:
                         child[idx] = np.clip(child[idx] + random.choice([-1, 1]), low, high)
                     else:
                         child[idx] = np.clip(child[idx] + random.randint(-5, 5), low, high)
                new_pop.append(child)
            population = new_pop
            
        return best_solution, best_prob

# --- 3. LOAD MODEL ---
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
st.write("Enter your full profile details to get a personalized strategy.")

with st.form("student_input"):
    col1, col2 = st.columns(2)
    
    # COLUMN 1: ACADEMICS & BASICS
    with col1:
        st.subheader("📚 Academics")
        ssc = st.number_input("SSC % (10th)", 40, 100, 85)
        hsc = st.number_input("HSC % (12th)", 40, 100, 85)
        degree = st.number_input("Degree %", 40, 100, 75)
        cgpa = st.number_input("Current CGPA", 0.0, 10.0, 8.0)
        backlogs = st.number_input("Current Backlogs", 0, 10, 0)
        attendance = st.slider("Attendance %", 0, 100, 80)
        entrance = st.number_input("Entrance Exam Score", 0, 100, 60)
        
    # COLUMN 2: SKILLS & EXPERIENCE
    with col2:
        st.subheader("🛠️ Skills & Experience")
        tech_score = st.slider("Technical Skill Score", 0, 100, 70)
        soft_score = st.slider("Soft Skill Score", 0, 100, 70)
        projects = st.slider("Live Projects", 0, 5, 1)
        internships = st.slider("Internships Completed", 0, 5, 1)
        work_exp = st.number_input("Work Experience (Months)", 0, 24, 0)
        certs = st.slider("Certifications", 0, 5, 1)
        extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    
    # GENDER (Hidden Default or Optional)
    gender_input = 1 # Male by default for simplicity, or add input if needed
    
    submitted = st.form_submit_button("Analyze My Profile")

# --- 5. EXECUTION LOGIC ---
if submitted and model is not None:
    # Convert Yes/No to 1/0
    extra_score = 1 if extra == "Yes" else 0

    input_data = {
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "degree_percentage": degree,
        "cgpa": cgpa,
        "technical_skill_score": tech_score,
        "soft_skill_score": soft_score,
        "internship_count": internships,
        "backlogs": backlogs,
        "live_projects": projects,
        "entrance_exam_score": entrance,
        "work_experience_months": work_exp,
        "certifications": certs,
        "attendance_percentage": attendance,
        "extracurricular_activities": extra_score,
        "gender": gender_input
    }
    
    df = pd.DataFrame([input_data])
    try:
        # Reorder columns to match model
        df = df[model_columns]
        student_vector = df.values[0]
        
        # DEFINING THE "ACTIONABLE" UNIVERSE
        # These are the things the AI is allowed to change to help you.
        constraints = {
            'technical_skill_score': (0, 100), 
            'soft_skill_score': (0, 100),
            'internship_count': (0, 5), 
            'backlogs': (0, 5), 
            'cgpa': (0, 10),
            'live_projects': (0, 5), 
            'entrance_exam_score': (0, 100),
            'work_experience_months': (0, 24), 
            'certifications': (0, 5),
            'extracurricular_activities': (0, 1)
        }
        actionable = list(constraints.keys())
        
        with st.spinner("AI is simulating 30+ future scenarios..."):
            opt = CareerOptimizer(model, model_columns, actionable, constraints)
            improved_vector, new_prob = opt.optimize(student_vector)
        
        # --- DISPLAY RESULTS ---
        st.divider()
        prob_percent = new_prob * 100
        
        # Gauge Chart Logic
        if prob_percent > 85:
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
                diff = improved - original
                
                # --- FILTERING LOGIC ---
                if abs(diff) < 0.5: continue
                # Don't suggest lowering skills
                if col != 'backlogs' and diff <= 0: continue
                # Don't suggest increasing backlogs
                if col == 'backlogs' and diff >= 0: continue

                # Text Formatting
                if col == 'backlogs':
                    msg = f"Clear {int(abs(diff))} Backlogs"
                elif col == 'extracurricular_activities':
                    msg = "Start participating in Extracurriculars"
                else:
                    msg = f"Increase by {round(diff, 1)}"

                changes_found = True
                with st.expander(f"Improve {col.replace('_', ' ').title()}"):
                    st.write(f"**Current:** {original} → **Target:** {round(improved, 1)}")
                    st.caption(f"👉 {msg}")

        # --- FINAL ANALYSIS MESSAGE ---
        if not changes_found:
            if prob_percent >= 85:
                st.balloons()
                st.success("🌟 You are a Top Candidate! Your profile is incredibly strong.")
            elif prob_percent > 50:
                st.info("ℹ️ Your profile is solid.")
                st.write(f"Your probability ({prob_percent:.1f}%) is good. Since your skills are optimized, the remaining gap is likely due to fixed past academic performance (SSC/HSC) which cannot be changed.")
            else:
                st.warning("⚠️ The AI couldn't find a single specific fix. This implies a need for holistic improvement across all areas (Projects, Internships, and Skills).")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Debug info: Ensure your CSV file has all the columns used in training.")