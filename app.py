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
        topic_logits = topic_model(**inputs).logits
        topic_idx = torch.argmax(topic_logits, dim=-1).item()
        topic = topic_labels.get(str(topic_idx), "Unknown")

        # Predict sentiment
        sentiment_logits = sentiment_model(**inputs).logits
        sentiment_idx = torch.argmax(sentiment_logits, dim=-1).item()
        sentiment = sentiment_labels.get(str(sentiment_idx), "Unknown")

        # Predict emotion
        emotion_logits = emotion_model(**inputs).logits
        emotion_idx = torch.argmax(emotion_logits, dim=-1).item()
        emotion = emotion_labels.get(str(emotion_idx), "Unknown")

    return topic, sentiment, emotion


def generate_therapeutic_response(user_input, topic, sentiment, emotion, conversation_stage):

    if conversation_stage == 0 or topic == "Unknown" or sentiment == "Unknown" or emotion == "Unknown":
        prompt = (
            "The user has provided limited information. "
            "Ask them open-ended questions to understand their feelings, concerns, or the specific issue they'd like to discuss."
        )
    elif conversation_stage == 1:
        prompt = (
            f"The user has shared feeling {sentiment} {emotion} about {topic}. Ask more detailed questions to uncover underlying concerns "
            f"or specific stressors. Focus on helping the user identify actionable steps to address their feelings."
        )
    elif conversation_stage == 2:
        prompt = (
            f"The user feels {sentiment} {emotion} about {topic}. Thank them for sharing. Introduce CBT or mindfulness techniques "
            f"to help them manage their anxiety effectively."
        )
    else:
        prompt = (
            "Wrap up the conversation by summarizing the user's concerns and offering further support. "
            "Ask if there's anything else they'd like to discuss."
        )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a therapeutic assistant specializing in anxiety support."},
            {"role": "user", "content": f"The user said: {user_input}. {prompt}"}
        ]
    )
    return response["choices"][0]["message"]["content"]


# Streamlit App Configuration
st.set_page_config(
    page_title="Anxiety Support Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("Anxiety Support Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = 0

def main():
    
    st.markdown(":gray[_*Type 'end session' anytime to close the conversation._]")

    # Display conversation history
    for message in st.session_state.messages:
        # Ensure message structure is correct
        role = message.get("role", "assistant")  # Default to "assistant" if "role" is missing
        content = message.get("content", "")  # Default to empty string if "content" is missing

        avatar_url = (
            "https://github.com/iisabelaaa/capstone/raw/main/user.png"
            if role == "user"
            else "https://github.com/iisabelaaa/capstone/raw/main/assistant.png"
        )

        try:
            with st.chat_message(role, avatar=avatar_url):
                st.markdown(content)
        except Exception as e:
            st.error(f"Error displaying message: {e}")
            continue

    # User input and response logic
    if prompt := st.chat_input("Welcome! I'm here to help you manage anxiety and provide support. What's on your mind?"):
        # Check for "end session" command
        if prompt.strip().lower() == "end session":
            st.session_state.clear()  # Clear the entire session state
            st.session_state.messages = []  # Ensure the messages list is reinitialized
            st.success("Session ended. Feel free to start a new conversation!")
            return  # Stop further execution for this interaction

        # Add user message to session
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="https://github.com/iisabelaaa/capstone/raw/main/user.png"):
            st.markdown(prompt)

        # Attempt to classify user input
        try:
            topic, sentiment, emotion = classify_sentiment_and_emotion(prompt)
            
            # Check if classification results are valid
            if topic == "Unknown" or sentiment == "Unknown" or emotion == "Unknown":
                clarification = (
                    "Thank you for reaching out! It seems I need a bit more detail to understand your situation. "
                    "Could you tell me more about how you're feeling or if there's something specific you'd like to discuss?"
                )
                with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
                    st.markdown(clarification)
                st.session_state.messages.append({"role": "assistant", "content": clarification})
                st.session_state.conversation_stage = 0
            else:
                # Generate assistant response
                assistant_response = generate_therapeutic_response(prompt, topic, sentiment, emotion, st.session_state.conversation_stage)
                with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
                    st.markdown(assistant_response)

                # Add assistant response to session
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                st.session_state.conversation_stage += 1  # Move to the next stage

        except Exception as e:
            with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
                st.error(f"An error occurred: {e}")

# Daisy Footer
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    text-align: center;
    padding: 10px 0;
}

.footer img {
    width: 100%;
    height: auto;
}

</style>
<div class="footer">
    <img src="https://raw.githubusercontent.com/iisabelaaa/capstone/main/daisy.png" alt="Daisy Footer">
</div>
"""
st.markdown(footer, unsafe_allow_html=True)


# Run App
if __name__ == "__main__":
    main()
