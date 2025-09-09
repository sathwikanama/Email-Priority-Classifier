import streamlit as st
import pickle
import tensorflow as tf
import numpy as np

# ---------- Load model & vectorizer ----------
model = tf.keras.models.load_model("email_priority_model.keras", compile=False)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Email Priority Classifier",
    page_icon="📌",
    layout="wide"
)

# ---------- Background style ----------
st.markdown("""
<style>
body { 
    background-image: url('https://images.unsplash.com/photo-1581091012184-1a2f3dc105b0?auto=format&fit=crop&w=1650&q=80'); 
    background-size: cover; 
    color: #fff; 
}
.stButton>button { 
    background-color: #1E90FF; 
    color: white; 
    height: 3em; 
    width: 100%; 
    border-radius:10px; 
}
.stTextArea>div>div>textarea { 
    border-radius:10px; 
}
.stTextInput>div>div>input { 
    border-radius:10px; 
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<h1 style='text-align:center; font-family:Arial Black;'>📌 AI Email Priority Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #f0f0f0;'>", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings ⚙️")
    threshold = st.slider("Importance Threshold (%)", 30, 90, 50)  # lowered default
    st.markdown("---")
    st.markdown("**Predicts if your email is:**")
    st.markdown("Important (green) ✅ ")
    st.markdown("**Or**")
    st.markdown("Not Important (red) ❌")

# ---------- Inputs ----------
subject = st.text_input("Email Subject", placeholder="Type email subject here...")
body = st.text_area("Email Body", height=150, placeholder="Type email body here...")

# ---------- Prediction ----------
if st.button("Check Priority 🚀"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter some text to check.")
    else:
        with st.spinner("Analyzing your email... ⏳"):
            combined_text = subject + " " + body
            email_vec = vectorizer.transform([combined_text]).toarray()
            pred_prob = float(model.predict(email_vec)[0][0])
            important_conf = pred_prob * 100
            not_important_conf = 100 - important_conf

        # ---------- Confidence Bars ----------
        st.markdown("<hr style='border:2px solid #f0f0f0;'>", unsafe_allow_html=True)
        st.subheader("Prediction Result 🎯")

        bar_style = """
        <style>
        .bar-container {width: 100%; background-color: #e0e0e0; border-radius: 25px; margin-bottom: 10px;}
        .bar-fill {height: 25px; border-radius: 25px; text-align: center; color: white; line-height: 25px;}
        </style>
        """
        st.markdown(bar_style, unsafe_allow_html=True)

        # Dynamic bar colors
        st.markdown(f"""
        <div class="bar-container">
            <div class="bar-fill" style="width: {important_conf}%; background: linear-gradient(to right, #00cc66, #009933);">
                ✅ Important: {important_conf:.2f}%
            </div>
        </div>
        <div class="bar-container">
            <div class="bar-fill" style="width: {not_important_conf}%; background: linear-gradient(to right, #ff4d4d, #cc0000);">
                ❌ Not Important: {not_important_conf:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------- Final Result ----------
        if important_conf > threshold:
            st.success(f"✅ Classified as Important ({important_conf:.2f}% confidence)")
        else:
            st.error(f"❌ Classified as Not Important ({not_important_conf:.2f}% confidence)")

# ---------- Footer ----------
st.markdown("<hr style='border:2px solid #f0f0f0;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Developed by N.Sathwika | AI/ML Project</p>", unsafe_allow_html=True)
