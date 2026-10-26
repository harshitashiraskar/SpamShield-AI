# 🛡️ SpamShield AI — NLP Ham/Spam Message Detection

## 📌 Project Overview
SpamShield AI is a machine learning application that classifies text messages as Spam or Ham (Non-Spam). It uses Natural Language Processing (NLP) and a trained Logistic Regression model to analyze messages.

## 🚀 Key Features
- Spam and Ham message classification
- Spam and Ham probability percentages
- Text analysis and message diagnostics
- Suspicious phrase and URL detection
- Downloadable CSV analysis report
- Interactive Streamlit web application

## 🛠️ Technologies Used
- Python
- Pandas
- Natural Language Processing (NLP)
- Scikit-learn
- CountVectorizer
- Logistic Regression
- NLTK
- Streamlit

## ⚙️ How to Run the Project

1. Clone or download this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Ensure `app.py`, `spam_model.pkl`, and `vectorizer.pkl` are in the same folder.
4. Run `streamlit run app.py` in your terminal.
5. Open the local URL displayed in your terminal.

## 📊 Machine Learning Workflow
1. Input a text message.
2. Preprocess the text.
3. Transform the message using the saved CountVectorizer.
4. Predict the message class using the trained Logistic Regression model.
5. Display probability estimates and text diagnostics.

## ⚠️ Disclaimer
Model probabilities are estimates, not guarantees. Always verify suspicious messages through official channels.

## 👩‍💻 Author
Harshita Shiraskar
