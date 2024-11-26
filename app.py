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

# Custom CSS for Styling
st.markdown(
    """
    <style>
    /* General Page Background */
    body {
        background-color: #eaf6fb;
    }

    /* Chat Container Styling */
    .chat-container {
        padding: 10px;
        background-color: #ffffff;
        border: 2px solid #cce7f0;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* User and Assistant Messages */
    .user-message {
        color: #3b5998; /* Soft blue for user messages */
        font-weight: bold;
    }
    .assistant-message {
        color: #1c5d99; /* Slightly darker blue for assistant */
    }

    /* Instruction Styling */
    .instruction-text {
        color: #8c8c8c; /* Gray color */
        font-size: 13px;
        text-align: center;
        margin-top: 10px;
    }

    /* Input Box and Button */
    .input-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    textarea {
        flex: 4;
        resize: none;
        height: 50px !important;
        border-radius: 8px;
        border: 1px solid #a0d2eb;
        padding: 10px;
    }
    button {
        flex: 1;
        background-color: #5db7de;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        cursor: pointer;
        padding: 10px;
        margin-left: 10px;
        transition: background-color 0.3s ease;
    }
    button:hover {
        background-color: #489dc5;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #8c8c8c;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    # Welcome Section
    st.title("Anxiety Support Chatbot")
    st.markdown("<p>Welcome! I'm here to help you manage anxiety and provide support.</p>", unsafe_allow_html=True)

    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    # Chat Display
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state["conversation"]:
        if message["role"] == "user":
            st.markdown(f"<p class='user-message'><strong>You:</strong> {message['content']}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='assistant-message'><strong>Assistant:</strong> {message['content']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Instruction Text (below chatbox)
    st.markdown(
        "<p class='instruction-text'>*Type <strong>'end session'</strong> anytime to close the conversation.*</p>",
        unsafe_allow_html=True,
    )

    # Input Box and Button
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_area(
                label="",
                placeholder="How can I help you today?",
                height=70,
                max_chars=300,
                key="input_area"
            )
        with col2:
            send_button = st.button("Send", use_container_width=True)

    # Handle Input and Response
    if send_button and user_input.strip():
        if user_input.lower() == "end session":
            st.session_state["conversation"] = []  # Clear conversation history
            st.success("Session ended. Feel free to start a new conversation!")
        else:
            try:
                # Call your existing classify and generate functions (defined elsewhere)
                topic, sentiment, emotion = classify_sentiment_and_emotion(user_input)
                assistant_response = generate_therapeutic_response(user_input, topic, sentiment, emotion)

                # Append user and assistant messages to the conversation
                st.session_state["conversation"].append({"role": "user", "content": user_input})
                st.session_state["conversation"].append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif send_button:
        st.warning("Please enter a valid input.")

    # Footer
    st.markdown("<div class='footer'>Developed by <strong>IAL</strong> in 2024.</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
