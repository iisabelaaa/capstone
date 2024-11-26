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

# CSS for Chatbox Styling
st.markdown(
    """
    <style>
    .chatbox {
        background-color: #A7C7E7; /* Pastel blue */
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 60%;
    }
    .user-box {
        background-color: #F3F3F3; /* Light grey */
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 60%;
        margin-left: auto;
        margin-right: 0;
    }
    .assistant-box {
        background-color: #E9F5FB; /* Lighter pastel blue */
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 60%;
        margin-right: auto;
        margin-left: 0;
    }
    .input-container {
        display: flex;
        align-items: center;
        background-color: white;
        border: 1px solid #CCC;
        border-radius: 10px;
        padding: 5px;
        margin-top: 10px;
    }
    .input-container textarea {
        flex-grow: 1;
        resize: none;
        border: none;
        padding: 10px;
        font-size: 1em;
        border-radius: 10px;
        outline: none;
    }
    .input-container button {
        background-color: #A7C7E7; /* Pastel blue */
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 10px;
        font-size: 1em;
        cursor: pointer;
        margin-left: 10px;
    }
    .input-container button:hover {
        background-color: #85A9D0; /* Slightly darker blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    # Welcome Section
    st.title("Anxiety Support Chatbot")
    st.markdown("Welcome! I'm here to help you manage anxiety and provide support.")
    st.markdown(
        """
        <p style="color: gray; font-size: 0.9em; margin-top: 10px; margin-left: 10px;">
            * Type "end session" anytime to close the conversation.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    # Display conversation
    for message in st.session_state["conversation"]:
        if message["role"] == "user":
            st.markdown(f"<div class='user-box'>You: {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-box'>Assistant: {message['content']}</div>", unsafe_allow_html=True)

    # Input container
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    user_input = st.text_area("", placeholder="How can I help you today?", height=50, max_chars=300, key="input_area")
    send_button = st.button("Send", key="send_button")
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle user input
    if send_button and user_input.strip():
        if user_input.lower() == "end session":
            st.session_state["conversation"] = []  # Clear the conversation history
            st.success("Session ended. Feel free to start a new conversation!")
        else:
            try:
                # Call your sentiment, emotion, and topic classification functions
                topic, sentiment, emotion = classify_sentiment_and_emotion(user_input)
                assistant_response = generate_therapeutic_response(user_input, topic, sentiment, emotion)

                # Append the conversation
                st.session_state["conversation"].append({"role": "user", "content": user_input})
                st.session_state["conversation"].append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif send_button:
        st.warning("Please enter a valid input.")

    # Footer
    st.write("---")
    st.markdown("<p style='text-align: center;'>Developed by IAL in 2024</p>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
