import streamlit as st
import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import login
import openai

# Access API keys from Streamlit secrets
huggingface_api_key = st.secrets["huggingface"]["api_key"]
openai_api_key = st.secrets["openai"]["api_key"]

# Log in to Hugging Face
login(huggingface_api_key)

# Set OpenAI API key
openai.api_key = openai_api_key

# Load label mappings from JSON
with open("label_mappings.json", "r") as f:
    label_mappings = json.load(f)

topic_labels = label_mappings["topic_labels"]
sentiment_labels = label_mappings["sentiment_labels"]
emotion_labels = label_mappings["emotion_labels"]

# Paths to model folders
topic_model = RobertaForSequenceClassification.from_pretrained("isabelaal/roberta_topic_classifier")
sentiment_model = RobertaForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
emotion_model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# Set models to evaluation mode
topic_model.eval()
sentiment_model.eval()
emotion_model.eval()

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Check device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in [topic_model, sentiment_model, emotion_model]:
    model.to(device)

# Function to classify sentiment, emotion, and topic
def classify_sentiment_and_emotion(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        # Predict topic
        topic_idx = torch.argmax(topic_model(**inputs).logits, dim=-1).item()
        topic = topic_labels.get(str(topic_idx), "Unknown")

        # Predict sentiment
        sentiment_idx = torch.argmax(sentiment_model(**inputs).logits, dim=-1).item()
        sentiment = sentiment_labels.get(str(sentiment_idx), "Unknown")

        # Predict emotion
        emotion_idx = torch.argmax(emotion_model(**inputs).logits, dim=-1).item()
        emotion = emotion_labels.get(str(emotion_idx), "Unknown")

    return topic, sentiment, emotion

# Function to generate therapeutic response using GPT-3.5
def generate_therapeutic_response(user_input, topic, sentiment, emotion):
    prompt = (
        f"The user feels {sentiment} {emotion} about {topic}. First talk to them and ask about {user_input}. "
        f"After they answer, provide a supportive response guided by Cognitive Behavioral Therapy and Mindfulness-Based Stress Reduction Therapy."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a therapeutic assistant specializing in anxiety support."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit App Configuration
st.set_page_config(
    page_title="Anxiety Support Chatbot",
    page_icon="🤖",
    layout="centered",
)

def main():
    # Welcome Section
    st.markdown(
        """
        <div style='background-color: #F0F8FF; padding: 10px; border-radius: 10px;'>
            <h1 style="text-align: center; color: #4682B4;">Anxiety Support Chatbot</h1>
            <p style="text-align: center; color: #6A5ACD;">Welcome! I'm here to help you manage anxiety and provide support.</p>
            <p style="text-align: center; font-style: italic; color: gray;">*Type <strong>'end session'</strong> anytime to close the conversation.*</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    # Display conversation
    with st.container():
        for message in st.session_state["conversation"]:
            if message["role"] == "user":
                st.markdown(
                    f"<div style='text-align: right; background-color: #BBDEFB; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align: left; background-color: #E3F2FD; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )

    # Input and Send Button Layout
    with st.form("user_input_form", clear_on_submit=True):
        user_input = st.text_area(
            "How can I help you today?",
            height=70,
            max_chars=300,
            placeholder="Type your message here...",
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        if user_input.strip():
            if user_input.lower() == "end session":
                st.session_state["conversation"] = []  # Clear the conversation history
                st.success("Session ended. Feel free to start a new conversation!")
            else:
                try:
                    # Classify and Generate
                    topic, sentiment, emotion = classify_sentiment_and_emotion(user_input)
                    assistant_response = generate_therapeutic_response(user_input, topic, sentiment, emotion)

                    # Append conversation history
                    st.session_state["conversation"].append({"role": "user", "content": user_input})
                    st.session_state["conversation"].append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid input.")

    # Footer
    st.write("---")
    st.markdown('<p style="text-align: center; color: gray;">Developed by <strong>IAL</strong> in 2024.</p>', unsafe_allow_html=True)

# Run App
if __name__ == "__main__":
    main()
