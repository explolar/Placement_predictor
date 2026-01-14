import streamlit as st
import requests
import json

# Set the page configuration
st.set_page_config(page_title="AI Career Coach", page_icon="🎓", layout="centered")

st.title("🎓 AI Career Placement Coach")
st.write("Enter your academic details to get a personalized placement strategy.")

# --- INPUT FORM ---
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

    # Hidden fields (defaults for now)
    live_projects = 0 
    work_exp = 0
    certifications = 0
    attendance = 75
    
    submitted = st.form_submit_button("Analyze My Profile")

# --- LOGIC ---
if submitted:
    # 1. Prepare the JSON payload matches your API Schema
    payload = {
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "degree_percentage": degree,
        "cgpa": cgpa,
        "technical_skill_score": tech_score,
        "soft_skill_score": soft_score,
        "internship_count": internships,
        "backlogs": backlogs,
        # Add default values for fields we didn't ask for to keep UI clean
        "live_projects": live_projects,
        "work_experience_months": work_exp,
        "certifications": certifications,
        "attendance_percentage": attendance,
        "entrance_exam_score": 50, # Dummy
        "student_id": 0, # Dummy
        "gender": 1, # Dummy
        "extracurricular_activities": 0 # Dummy
    }

    try:
        # 2. Call the FastAPI Backend
        response = requests.post("http://127.0.0.1:8000/advise", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # 3. Display Results
            st.divider()
            
            # Probability Gauge
            prob_str = result['achievable_probability'].replace('%', '')
            prob = float(prob_str)
            
            if prob > 80:
                st.success(f"✅ High Chance of Placement: {result['achievable_probability']}")
            elif prob > 50:
                st.warning(f"⚠️ Moderate Chance: {result['achievable_probability']}")
            else:
                st.error(f"❌ Low Chance: {result['achievable_probability']}")
                
            st.subheader("🚀 Your Action Plan")
            
            if not result['action_plan']:
                st.info("Your profile is already optimized! Keep it up.")
            else:
                for item in result['action_plan']:
                    with st.expander(f"Improve {item['feature'].replace('_', ' ').title()}"):
                        st.write(f"**Current:** {item['current']}")
                        st.write(f"**Target:** {item['target']}")
                        st.write(f"👉 **{item['action']}**")
        else:
            st.error("Error connecting to the AI brain.")
            st.write(response.text)
            
    except Exception as e:
        st.error(f"Connection Failed. Is the backend running? {e}")