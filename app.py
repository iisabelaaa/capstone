import streamlit as st
import torch
import openai
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from huggingface_hub import login
login("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")

# Set your OpenAI API key
openai.api_key = "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"

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
    st.title("Anxiety Support Chatbot")
    st.markdown("Welcome! I'm here to help you manage anxiety and provide support.")
    st.markdown("**Type 'end session' anytime to close the conversation.**")

    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    # Display conversation
    for message in st.session_state["conversation"]:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

    # User Input
    user_input = st.text_input("How can I help you today?", "")

    if st.button("Send"):
        if user_input.strip():
            if user_input.lower() == "end session":
                st.session_state["conversation"] = []  # Clear the conversation history
                st.success("Session ended. Feel free to start a new conversation!")
            else:
                try:
                    # Call functions defined in another cell
                    topic, sentiment, emotion = classify_sentiment_and_emotion(user_input)
                    assistant_response = generate_therapeutic_response(user_input, topic, sentiment, emotion)

                    # Append the interaction to the conversation
                    st.session_state["conversation"].append({"role": "user", "content": user_input})
                    st.session_state["conversation"].append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid input.")

    # Footer
    st.write("---")
    st.markdown("Developed by IAL in 2024.")

# Run the app
if __name__ == "__main__":
    main()
