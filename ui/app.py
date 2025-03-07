import streamlit as st
import requests

# Title
st.set_page_config(page_title="Custom Transformer AI", layout="centered")
st.title("🤖 Custom Transformer AI")

# Description
st.markdown("This AI generates responses based on a custom-trained Transformer model.")

# Input field for user prompt
user_input = st.text_area("💬 Enter your prompt:", "", height=150)

# Sidebar for settings
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/generate")
max_length = st.sidebar.slider("Max Response Length", min_value=20, max_value=200, value=50)

# Generate button
if st.button("🚀 Generate Response"):
    if user_input:
        try:
            response = requests.get(api_url, params={"prompt": user_input, "max_length": max_length})
            if response.status_code == 200:
                st.subheader("📝 AI Response:")
                st.success(response.json().get("generated_text", "No response received."))
            else:
                st.error(f"Error: Unable to get response from AI. Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
    else:
        st.warning("⚠️ Please enter a prompt before generating a response.")

st.markdown("---")
st.markdown("### 🔥 Custom AI built from scratch using Transformers.")