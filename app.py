import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("smartspam_svm_model.pkl")
vectorizer = joblib.load("smartspam_tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Page configuration
st.set_page_config(
    page_title="SmartSpam",
    page_icon="🛡️",
    layout="wide"
)


# Custom CSS
st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown(
    '<div class="main-title">🛡️ SmartSpam</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent SMS Spam Classification Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# Model information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 Model", "SVM")

with col2:
    st.metric("🎯 Accuracy", "97.68%")

with col3:
    st.metric("🧠 Features", "TF-IDF")


st.divider()


# Main section
left, right = st.columns([2, 1])


# Left side
with left:

    st.subheader("📱 Analyze Your SMS")

    st.write(
        "Enter an SMS message below and SmartSpam will "
        "classify it as Spam or Legitimate."
    )

    message = st.text_area(
        "SMS Message",
        height=180,
        placeholder="Type or paste your SMS here..."
    )

    if message:
        st.caption(f"Character count: {len(message)}")

    check_button = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )


# Right side
with right:

    st.subheader("🧪 Try Examples")

    st.write("Use these sample messages to test the model.")

    normal_example = st.button(
        "💬 Normal SMS",
        use_container_width=True
    )

    spam_example = st.button(
        "🚨 Spam SMS",
        use_container_width=True
    )


# Example messages
if normal_example:

    st.info(
        "Example:\n\n"
        "Hey, are you coming to college tomorrow?"
    )

if spam_example:

    st.info(
        "Example:\n\n"
        "Congratulations! You have won a free prize "
        "of 50000. Call now to claim your reward!"
    )


# Prediction
if check_button:

    if message.strip() == "":
        st.warning("⚠️ Please enter an SMS message first.")

    else:

        with st.spinner("🤖 Analyzing message..."):

            cleaned_message = clean_text(message)

            message_tfidf = vectorizer.transform(
                [cleaned_message]
            )

            prediction = model.predict(message_tfidf)[0]


        st.divider()

        if prediction == "spam":

            st.error("🚨 SPAM MESSAGE")

            st.write(
                "SmartSpam has classified this message as **Spam**."
            )

            st.warning(
                "⚠️ Be careful with messages containing "
                "prize claims, suspicious links, OTP requests "
                "or requests for money."
            )

        else:

            st.success("✅ LEGITIMATE MESSAGE")

            st.write(
                "SmartSpam has classified this message as "
                "**Ham / Legitimate**."
            )


# Footer
st.divider()

st.caption(
    "SmartSpam | TF-IDF + Support Vector Machine | "
    "UCI SMS Spam Collection"
)