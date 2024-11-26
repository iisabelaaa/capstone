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
def generate_therapeutic_response(user_input, topic, sentiment, emotion, conversation_stage):
    """
    Generate a therapeutic response based on user input, topic, sentiment, emotion, 
    and the current stage of the conversation.
    """
    # Create different prompts for different stages of the conversation
    if conversation_stage == 0:  # Initial stage: Understand the user's feelings
        prompt = (
            f"You are a therapeutic assistant specializing in anxiety support. The user feels {sentiment} {emotion} about {topic}. "
            "Start by asking them to share more about their feelings and what's causing this reaction."
        )
    elif conversation_stage == 1:  # Second stage: Explore specific concerns
        prompt = (
            f"You are a therapeutic assistant specializing in anxiety support. The user has shared their feelings. "
            f"Ask them more specific questions to help explore any underlying worries, fears, or negative thoughts. "
            f"Continue the conversation by guiding them to identify these specific concerns."
        )
    elif conversation_stage == 2:  # Third stage: Introduce CBT and support
        prompt = (
            f"You are a therapeutic assistant specializing in anxiety support. The user feels {sentiment} {emotion} about {topic}. "
            "Thank them for sharing their concerns. Introduce a Cognitive Behavioral Therapy technique like identifying negative thoughts, "
            "challenging unhelpful beliefs, or mindfulness exercises. Offer practical strategies to manage anxiety."
        )
    else:  # Default fallback: Supportive response
        prompt = (
            f"You are a therapeutic assistant specializing in anxiety support. The user feels {sentiment} {emotion} about {topic}. "
            "Provide a supportive, empathetic response and ask if there is anything else they’d like to share or work through."
        )

    # Call GPT-3.5 for generating the response
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

# Define custom CSS
custom_css = f"""
<style>
    body {{
        background-color: #CAF0F8; /* Pastel blue background */
        background-image: url('https://raw.githubusercontent.com/iisabelaaa/capstone/main/daisy_bg.jpg'); /* Daisy field image */
        background-repeat: no-repeat;
        background-position: bottom center;
        background-size: cover; /* Adjust size to fit */
    }}
    .stApp {{
        background: transparent; /* Ensures the app area remains clear */
    }}
</style>
"""

# Inject CSS into the app
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Anxiety Support Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = 0  # Start at the first stage

def main():
    # Add welcome message
    st.markdown(":gray[_*Type 'end session' anytime to close the conversation._]")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input and response logic
    if prompt := st.chat_input("Welcome! I'm here to help you manage anxiety and provide support. What's on your mind?"):
        # Add user message to session
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if the user wants to end the session
        if prompt.lower() == "end session":
            st.session_state.conversation_stage = 0  # Reset conversation stage
            st.session_state.messages = []  # Clear message history
            st.success("Session ended. Feel free to start a new conversation!")
        else:
            # Generate assistant response based on the conversation stage
            topic, sentiment, emotion = classify_sentiment_and_emotion(prompt)
            assistant_response = generate_therapeutic_response(
                prompt, topic, sentiment, emotion, st.session_state.conversation_stage
            )

            # Add assistant response to chat
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            # Increment the conversation stage
            st.session_state.conversation_stage += 1

# Run App
if __name__ == "__main__":
    main()
