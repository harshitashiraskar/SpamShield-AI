# SpamShield AI — NLP Ham/Spam Detection

## Project Overview
SpamShield AI is a machine learning application that classifies text messages as Spam or Ham (Non-Spam) using Natural Language Processing and Logistic Regression.

## Features
- Spam and Ham message classification
- Spam and Ham probability percentages
- Suspicious phrase and URL detection
- Message diagnostics
- Downloadable CSV analysis report
- Interactive Streamlit web application

## Technologies Used
- Python
- Pandas
- NLP
- Scikit-learn
- CountVectorizer
- Logistic Regression
- NLTK
- Streamlit

## How to Run
1. Install the required packages using `pip install -r requirements.txt`.
2. Keep `app.py`, `spam_model.pkl`, and `vectorizer.pkl` in the same folder.
3. Run `streamlit run app.py` in your terminal.

## Disclaimer
Predictions are estimates and do not guarantee whether a message is safe or malicious. Verify suspicious messages through official channels.

## Author
Harshita Shiraskar