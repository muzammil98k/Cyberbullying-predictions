# 🚫 AI-Augmented System for Identifying Online Harassment and Cyberbullying

This project focuses on building a machine learning-based system to detect and classify instances of online harassment and cyberbullying. It combines Natural Language Processing (NLP) techniques with classification algorithms to identify harmful text and flag abusive behavior based on categories such as gender, religion, ethnicity, and more.

## 🔍 Project Objective

To develop an AI-augmented platform that:
- Detects and classifies cyberbullying content in real-time
- Provides data analysis based on abuse categories
- Supports prediction and moderation tools for safer online interactions

## 📊 Features

- Text preprocessing & tokenization
- Word cloud generation for category-wise abuse
- Multi-model classification (Random Forest, AdaBoost, XGBoost)
- Data visualization with Matplotlib & Seaborn
- Flask web application for user interaction

## 🧠 ML Models Used

- **Random Forest**
- **AdaBoost**
- **XGBoost**
- **Logistic Regression (baseline)**

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas, NumPy**
- **NLTK / SpaCy**
- **Matplotlib, Seaborn**
- **HTML/CSS (Frontend UI)**

## 🧪 Dataset

The dataset includes user-generated text from social media and forums, labeled with various abuse categories such as:
- Gender
- Religion
- Ethnicity
- Nationality

> **Note:** Dataset is anonymized and preprocessed for ethical compliance.

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/cyberbullying-detector.git
cd cyberbullying-detector

# Create a virtual environment and install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
