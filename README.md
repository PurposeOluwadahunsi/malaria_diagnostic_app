# Malaria Risk Detector

A machine learning-powered web app that estimates the probability of malaria based on patient symptoms. Built with Streamlit and a Gradient Boosting Classifier trained on a Kaggle malaria dataset.

> **Disclaimer:** This tool is a probability estimator, not a medical diagnosis. Always visit a certified healthcare facility for proper testing and treatment.

## Live App

[https://malaria-diagnosis-system.streamlit.app](https://malaria-diagnosis-system.streamlit.app)

---

## Problem Statement

In Nigeria and many parts of Africa, people often self-diagnose malaria when they feel common symptoms like fever or headaches, without getting properly tested. This can lead to wrong treatments or delayed care. This app gives users a data-driven hint about their risk level so they can make a more informed decision about seeking medical help.

---

## How It Works

The user inputs:
- Age and sex
- Yes/No answers to 11 symptoms: Fever, Headache, Abdominal Pain, General Body Malaise, Dizziness, Vomiting, Confusion, Backache, Chest Pain, Coughing, Joint Pain

The app returns:
- A prediction: **Malaria** or **No Malaria**
- A probability score (0–100%)
- A risk level: **Low**, **Medium**, or **High**
- A battery-style visual showing the probability

---

## Model Development

**Dataset:** Malaria dataset from Kaggle

**Preprocessing:**
- Dropped columns that could cause data leakage or were irrelevant: `IP Number`, `DOA`, `Discharge Date`, `Primary Code`, `Diagnosis Type`, `Risk Score`
- Encoded the `Sex` column (Male = 1, Female = 0)

**Models trained and their accuracy:**

| Model | Accuracy |
|---|---|
| Logistic Regression | 96% |
| Random Forest Classifier | 97% |
| Gradient Boosting Classifier | **99%** |

**Final model:** Gradient Boosting Classifier (saved as `Malaria_Diagnostic_Model.pkl`)

---

## Features

- Clean, mobile-friendly Streamlit UI
- Battery-style probability display
- Progress animation during analysis
- Feedback form connected to Supabase (users can report whether the prediction matched their actual clinic result)
- Session state management to preserve results across reruns

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NumPy
- Supabase for feedback storage 

---

## Project Structure

