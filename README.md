# 🎓 AI Career Placement Coach & Counterfactual Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://placementpredictor-irbrr8lqtnarybgk2d84xr.streamlit.app/)

> **Don't just predict failure. Fix it.**

## 🔗 Live Demo
👉 **[Click here to try the App](https://placementpredictor-irbrr8lqtnarybgk2d84xr.streamlit.app/)**

---

## 📖 Overview
This is not just a standard "Placement Prediction" model. It is an **Actionable AI System** that predicts a student's probability of getting placed and, if they are at risk, uses a **Genetic Algorithm (Evolutionary Search)** to generate a personalized, realistic "Counterfactual Plan" to turn that failure into success.

**The Core Question it solves:** *"I know I might fail, but exactly what do I need to change in my profile to pass?"*

---

## 🚀 Features
- **🔮 Prediction Engine**: Uses a **Random Forest Classifier** (trained on `student_academic_placement_performance_dataset.csv`) to predict placement status with high accuracy.
- **🧬 Counterfactual Optimizer**: A custom **Genetic Algorithm** that evolves a student's resume features (Skills, Backlogs, Internships) to find the *easiest* path to a 90%+ placement probability.
- **⚡ Fast Inference**: optimized for real-time suggestions.
- **🖥️ Interactive Dashboard**: A user-friendly **Streamlit** frontend for students to input their data and visualize their "Action Plan."

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Optimization**: Custom Genetic Algorithm (Evolutionary Strategy)
- **Deployment**: Streamlit Cloud

---

## 📂 Project Structure
```bash
D:\pl_pre\
│
├── student_academic_placement_performance_dataset.csv  # The Raw Data
├── train_model.py    # Script to train ML model & save artifacts
├── optimizer.py      # The Brain (Genetic Algorithm Logic)
├── main.py           # The API (FastAPI Backend - Optional if Monolithic)
├── app.py            # The UI (Streamlit Dashboard)
├── model.pkl         # Saved Model (Generated after training)
├── columns.pkl       # Saved Column Names (Generated after training)
└── README.md         # Project Documentation
