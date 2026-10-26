
import streamlit as st
import pandas as pd
import pickle
import string
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from datetime import datetime

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SpamShield AI | Message Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. MODEL FILE LOCATIONS
# =========================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "spam_model.pkl"
VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"

# =========================================================
# 3. TEXT PREPROCESSING
# IMPORTANT: Keep this function name because your saved
# CountVectorizer uses the custom analyzer text_process.
# =========================================================
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
    except Exception as e:
        st.error(
            "NLTK stopwords are missing. Run this command "
            "in your terminal:\n\n"
            "python -m nltk.downloader stopwords"
        )
        st.stop()


def text_process(message):
    message = str(message)
    nopunc = [
        char for char in message
        if char not in string.punctuation
    ]
    nopunc = "".join(nopunc)

    return [
        word for word in nopunc.split()
        if word not in STOP_WORDS
    ]


# =========================================================
# 4. LOAD TRAINED MODEL AND VECTORIZER
# =========================================================
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


try:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not VECTORIZER_PATH.exists():
        st.error(f"Vectorizer file not found: {VECTORIZER_PATH}")
        st.stop()

    model, vectorizer = load_artifacts()

except Exception as e:
    st.error("Unable to load the trained model.")
    st.write(
        "Check that both pickle files are in the same folder "
        "as app.py and that the required Python packages "
        "and training function are available."
    )
    st.code(str(e))
    st.stop()


# =========================================================
# 5. CUSTOM COLORFUL DESIGN
# =========================================================
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #f4f8ff 0%,
        #e9f7ff 55%,
        #effcf7 100%
    );
    color: #172b4d;
}

/* Content width and spacing */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main headings */
.stMain h1 {
    color: #102a56 !important;
    font-size: 2.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

.stMain h2 {
    color: #123b67 !important;
    font-size: 1.8rem !important;
    font-weight: 750 !important;
}

.stMain h3 {
    color: #087f8c !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}

/* Main content readability */
.stMain p,
.stMain li,
.stMain label {
    color: #334766;
    line-height: 1.65;
}

/* Dark navy sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #102447 0%,
        #163c63 65%,
        #075b68 100%
    );
    border-right: 1px solid #2b5878;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] label {
    color: #e2ecf8 !important;
    line-height: 1.6;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #75f0df !important;
    font-weight: 750 !important;
}

/* Sidebar separators */
section[data-testid="stSidebar"] hr {
    border-color: #52718c !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #172b4d !important;
    border: 1px solid #b9d8e8 !important;
    border-radius: 12px !important;
    line-height: 1.6 !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #172b4d !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        100deg,
        #087f8c,
        #00a6a6
    ) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    padding: 0.65rem 1.15rem !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px #087f8c25;
}

.stButton > button p {
    color: #ffffff !important;
}

.stButton > button:hover {
    background: linear-gradient(
        100deg,
        #075b68,
        #087f8c
    ) !important;
    transform: translateY(-2px);
    box-shadow: 0 7px 18px #087f8c35;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d8e8f4;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 4px 14px #15365b0b;
}

div[data-testid="stMetricLabel"] {
    color: #526987 !important;
}

div[data-testid="stMetricValue"] {
    color: #087f8c !important;
    font-weight: 800 !important;
}

/* Expander cards */
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #d8e8f4;
    border-radius: 12px;
}

/* Captions */
.stCaption {
    color: #607795 !important;
    font-size: 0.88rem !important;
}

/* Download button */
.stDownloadButton > button {
    background: #102a56 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

/* Sidebar button contrast */
section[data-testid="stSidebar"] .stButton > button {
    background: #e0fffa !important;
    color: #103a52 !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    color: #103a52 !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Reduce spacing around the page header */
.stMain h1 {
    margin-bottom: 0.25rem !important;
}

.stMain h2 {
    margin-top: 0.25rem !important;
    margin-bottom: 0.4rem !important;
}

/* Reduce space below the description */
.stMain div[data-testid="stMarkdown"] p {
    margin-bottom: 0.35rem !important;
}

/* Tighten the gap around dividers */
.stMain hr {
    margin-top: 0.6rem !important;
    margin-bottom: 0.8rem !important;
}


/* Prevent the top content from being clipped */
.stMain .block-container {
    padding-top: 2.5rem !important;
}

/* Keep the top toolbar positioned correctly */
[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 1000;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 6. SAMPLE MESSAGES
# =========================================================
SAMPLES = {
    "📩 Select a sample message": "",
    "✅ Normal message - meeting": (
        "Hi, are we still meeting at 5 pm today? "
        "Please let me know."
    ),
    "✅ Normal message - friend": (
        "Hey, I reached home safely. Thank you for "
        "checking on me."
    ),
    "🚨 Suspicious message - prize": (
        "CONGRATULATIONS! You have won a FREE prize. "
        "Claim your cash reward now by clicking this link!"
    ),
    "🚨 Suspicious message - urgent account": (
        "URGENT! Your bank account will be blocked today. "
        "Verify your account immediately by clicking this link."
    ),
    "🚨 Suspicious message - lottery": (
        "You have won 500000 dollars in our lottery. "
        "Send your bank details now to receive your prize."
    )
}


def load_selected_sample():
    selected = st.session_state.get("sample_choice", "")
    st.session_state["message_input"] = SAMPLES.get(selected, "")


def clear_message():
    st.session_state["message_input"] = ""
    st.session_state["analysis_result"] = None


if "message_input" not in st.session_state:
    st.session_state["message_input"] = ""

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if "sample_choice" not in st.session_state:
    st.session_state["sample_choice"] = "📩 Select a sample message"


# =========================================================
# 7. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🛡️ SpamShield AI")
    st.caption("SMART MESSAGE PROTECTION")

    st.divider()

    st.subheader("📌 About the Project")
    st.write(
        "An NLP-based message classifier that identifies "
        "potential spam and helps users review suspicious texts."
    )

    st.subheader("📚 Understanding the Results")

    st.markdown(
        "**✅ HAM — Non-spam**\n\n"
        "The model classifies the message as a normal message."
    )

    st.markdown(
        "**🚨 SPAM — Potential spam**\n\n"
        "The model classifies the message as spam."
    )

    st.divider()

    st.subheader("⚙️ Technology Stack")
    st.markdown(
        "- 🐍 Python\n"
        "- 🧠 Natural Language Processing\n"
        "- 📊 Scikit-learn\n"
        "- 🎈 Streamlit\n"
        "- 🤖 Logistic Regression"
    )

    st.divider()

    st.subheader("🔍 What You Can Explore")
    st.markdown(
        "- Message classification\n"
        "- Model probability scores\n"
        "- Text characteristics\n"
        "- Suspicious keyword checks\n"
        "- Downloadable analysis report"
    )

    st.divider()

    st.subheader("🔐 Safety Reminder")
    st.write(
        "Predictions are not guarantees. Avoid clicking "
        "unknown links or sharing passwords, OTPs, or "
        "banking details. Verify important messages "
        "through official channels."
    )

    st.divider()
    st.caption("Built with Python, NLP & Machine Learning ✨")


# =========================================================
# 8. MAIN PAGE HEADER
# =========================================================
st.caption("🔎 INTELLIGENT MESSAGE SCREENING")

st.title("🛡️ SpamShield AI")

st.subheader("Don't Trust Every Text. Verify It. ✨")

# st.write(
#     "Analyze a message, detect potential spam, and review "
#     "the model's prediction before deciding whether to trust it."
# )

st.markdown("""
<div style="
    font-size: 18px;
    line-height: 1.8;
    color: #334766;
">
Analyze a message, detect potential spam, and review the model's prediction<br>
before deciding whether to trust it.
</div>
""", unsafe_allow_html=True)


st.divider()


# =========================================================
# 9. MESSAGE INPUT
# =========================================================
st.header("💬 Analyze a Message")

st.caption(
    "Paste a text message below or choose a sample "
    "to try the classifier."
)

st.selectbox(
    "🧪 Try an example",
    options=list(SAMPLES.keys()),
    key="sample_choice",
    on_change=load_selected_sample
)

message = st.text_area(
    "📝 Enter the message you want to check",
    key="message_input",
    height=155,
    placeholder=(
        "Example: Congratulations! You have won a prize. "
        "Click here to claim it..."
    ),
    help="Paste the message text. Avoid including real passwords or OTPs."
)

col_analyze, col_clear, col_hint = st.columns([1.2, 0.8, 2])

with col_analyze:
    analyze_clicked = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )

with col_clear:
    st.button(
        "🧹 Clear",
        on_click=clear_message,
        use_container_width=True
    )

with col_hint:
    st.caption("💡 Try both normal and suspicious examples.")


# =========================================================
# 10. ANALYSIS FUNCTIONS
# =========================================================
def inspect_message(message):
    words = message.split()

    urls = re.findall(
        r"(?:https?://\S+|www\.\S+)",
        message,
        flags=re.IGNORECASE
    )

    emails = re.findall(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        message
    )

    uppercase_letters = [
        char for char in message if char.isalpha()
    ]

    uppercase_percentage = (
        sum(char.isupper() for char in uppercase_letters)
        / len(uppercase_letters) * 100
        if uppercase_letters else 0
    )

    suspicious_terms = [
        "winner", "won", "prize", "free", "claim",
        "urgent", "click", "cash", "reward",
        "verify", "blocked", "lottery", "offer",
        "congratulations", "bank details", "password",
        "limited time", "act now"
    ]

    message_lower = message.lower()

    found_terms = [
        term for term in suspicious_terms
        if term in message_lower
    ]

    return {
        "Characters": len(message),
        "Words": len(words),
        "URLs detected": len(urls),
        "Email addresses detected": len(emails),
        "Uppercase letters": f"{uppercase_percentage:.1f}%",
        "Suspicious phrases found": (
            ", ".join(found_terms) if found_terms else "None detected"
        ),
        "URLs found": ", ".join(urls) if urls else "None detected"
    }


def analyze_message(message):
    # Match the lowercase preprocessing used during training.
    message_lower = message.lower()

    # Use the SAME saved CountVectorizer.
    message_vector = vectorizer.transform([message_lower])

    # Prediction: 0 = ham, 1 = spam.
    prediction = model.predict(message_vector)[0]

    spam_probability = None
    ham_probability = None

    # Get probabilities when the fitted model supports them.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(message_vector)[0]
        classes = list(model.classes_)

        if 1 in classes:
            spam_probability = float(probabilities[classes.index(1)])

        if 0 in classes:
            ham_probability = float(probabilities[classes.index(0)])

    # A model without predict_proba can still provide its class.
    return {
        "prediction": int(prediction),
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
        "details": inspect_message(message)
    }


# =========================================================
# 11. RUN THE ANALYSIS
# =========================================================
if analyze_clicked:
    if not message.strip():
        st.warning(
            "✍️ Please enter a message before clicking Analyze."
        )
        st.session_state["analysis_result"] = None
    else:
        try:
            with st.spinner("🧠 Analyzing message..."):
                result = analyze_message(message.strip())

            st.session_state["analysis_result"] = {
                **result,
                "message": message.strip(),
                "analyzed_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }

        except Exception as e:
            st.error("Something went wrong during analysis.")
            st.code(str(e))


# =========================================================
# 12. DISPLAY THE RESULTS
# =========================================================
result = st.session_state.get("analysis_result")

if result:
    st.divider()
    st.header("📊 Analysis Results")

    is_spam = result["prediction"] == 1
    spam_prob = result["spam_probability"]
    ham_prob = result["ham_probability"]

    if is_spam:
        st.error(
            "🚨 SPAM DETECTED — The model classified this "
            "message as potential spam."
        )
    else:
        st.success(
            "✅ NON-SPAM (HAM) — The model classified this "
            "message as non-spam."
        )

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "that a message is safe or malicious."
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "🏷️ Predicted Class",
            "SPAM 🚨" if is_spam else "HAM ✅"
        )

    with metric2:
        if spam_prob is not None:
            st.metric(
                "🚨 Spam Probability",
                f"{spam_prob:.1%}"
            )
        else:
            st.metric("🚨 Spam Probability", "Unavailable")

    with metric3:
        if ham_prob is not None:
            st.metric(
                "✅ Ham Probability",
                f"{ham_prob:.1%}"
            )
        else:
            st.metric("✅ Ham Probability", "Unavailable")

    if spam_prob is not None and ham_prob is not None:
        st.subheader("📈 Model Probability Breakdown")

        st.write("🚨 Spam")
        st.progress(max(0.0, min(1.0, spam_prob)))

        st.write("✅ Ham / Non-spam")
        st.progress(max(0.0, min(1.0, ham_prob)))

        st.caption(
            "These values represent the model's estimated class "
            "probabilities. They are not guaranteed measures of "
            "real-world risk or certainty."
        )

    st.divider()

    # Message reviewed
    st.subheader("📝 Message Reviewed")
    st.info(result["message"])

    # Text diagnostics
    st.subheader("🔬 Message Diagnostics")

    details = result["details"]

    diag1, diag2, diag3 = st.columns(3)

    with diag1:
        st.metric("🔤 Characters", details["Characters"])

    with diag2:
        st.metric("🗣️ Words", details["Words"])

    with diag3:
        st.metric("🔗 URLs", details["URLs detected"])

    with st.expander("🔍 View detailed text checks"):
        st.write(
            f"**Email addresses:** "
            f"{details['Email addresses detected']}"
        )
        st.write(
            f"**Uppercase letters:** "
            f"{details['Uppercase letters']}"
        )
        st.write(
            f"**Suspicious phrases:** "
            f"{details['Suspicious phrases found']}"
        )
        st.write(
            f"**URLs found:** {details['URLs found']}"
        )

    # Downloadable report
    st.divider()
    st.subheader("📥 Export Your Analysis")

    report = pd.DataFrame([{
        "Analyzed At": result["analyzed_at"],
        "Message": result["message"],
        "Predicted Class": "SPAM" if is_spam else "HAM",
        "Spam Probability": (
            round(spam_prob, 4) if spam_prob is not None else None
        ),
        "Ham Probability": (
            round(ham_prob, 4) if ham_prob is not None else None
        ),
        "Characters": details["Characters"],
        "Words": details["Words"],
        "URLs Detected": details["URLs detected"],
        "Email Addresses Detected": (
            details["Email addresses detected"]
        ),
        "Suspicious Phrases": details["Suspicious phrases found"]
    }])

    st.download_button(
        label="📄 Download Analysis Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8-sig"),
        file_name="spamshield_analysis.csv",
        mime="text/csv",
        use_container_width=False
    )


# =========================================================
# 13. HOW IT WORKS
# =========================================================
st.divider()
st.header("⚙️ How It Works")

st.caption(
    "A simple overview of the machine-learning workflow."
)

step1, step2, step3 = st.columns(3)

with step1:
    st.subheader("1️⃣ Input")
    st.write(
        "Provide a text message for the classifier to examine."
    )

with step2:
    st.subheader("2️⃣ Process")
    st.write(
        "The saved CountVectorizer transforms the message "
        "into numerical features."
    )

with step3:
    st.subheader("3️⃣ Predict")
    st.write(
        "The trained Logistic Regression model predicts "
        "the message class."
    )

st.divider()

st.caption(
    "🛡️ SpamShield AI | NLP & Machine Learning Portfolio Project"
)








 