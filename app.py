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
    # Handle stage -1 for unknown classifications
    if conversation_stage == -1:
        prompt = (
            "The classification of the user's input was unclear. Greet them warmly and encourage them to share more about their feelings "
            "or what’s on their mind. Avoid making any assumptions."
        )
    elif conversation_stage == -2:
        prompt = (
            "The user has indicated they want emotional support without strategies. Respond empathetically, "
            "validate their feelings, and focus on being a comforting presence without suggesting solutions."
        )
    elif conversation_stage == 0:
        prompt = (
            "The user is starting the conversation. Greet them warmly and encourage them to share their thoughts and feelings."
        )
    elif conversation_stage == 1:
        prompt = (
            f"The user feels {emotion} and {sentiment} about {topic}. Respond empathetically and ask an open-ended question to explore the situation further."
        )
    elif conversation_stage == 2:
        prompt = (
            f"The user has shared feeling {emotion} and {sentiment} about {topic}. Ask them to identify specific aspects causing their anxiety or stress. "
            "Focus on understanding their main triggers."
        )
    elif conversation_stage == 3:
        prompt = (
            f"The user is feeling {emotion} and {sentiment} about {topic}. They mentioned specific concerns. Suggest actionable CBT techniques, "
            "such as thought challenging, reframing, or small preparation steps. Offer examples if needed."
        )
    elif conversation_stage == 4:
        prompt = (
            f"The user is feeling {emotion} and {sentiment} about {topic}. They want detailed guidance on applying CBT techniques. "
            "Explain how to use CBT methods step by step, like thought records, exposure tasks, or mindfulness exercises. Provide clear examples."
        )
    elif conversation_stage == 5:
        prompt = (
            f"The user has discussed their concerns about {topic} and seems to be wrapping up. Reinforce their progress and offer encouragement. "
            "If they request more strategies, redirect to earlier stages for additional techniques. If they bring up a new topic, restart at stage 1."
        )
    else:
        prompt = (
            "Summarize the key points discussed, express encouragement, and offer additional support. Redirect the user if they wish to explore something new or need more help."
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

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_stage" not in st.session_state:
        st.session_state.conversation_stage = 0

    st.markdown(":gray[_*Type 'end session' anytime to close the conversation._]")

    # Display conversation history
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        avatar_url = (
            "https://github.com/iisabelaaa/capstone/raw/main/user.png"
            if role == "user"
            else "https://github.com/iisabelaaa/capstone/raw/main/assistant.png"
        )

        with st.chat_message(role, avatar=avatar_url):
            st.markdown(content)

    # User input and response logic
    if prompt := st.chat_input("Welcome! I'm here to help you manage anxiety and provide support. What's on your mind?"):
        if prompt.strip().lower() == "end session":
            st.session_state.clear()
            st.session_state.messages = []
            st.session_state.conversation_stage = 0
            st.success("Session ended. Feel free to start a new conversation!")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="https://github.com/iisabelaaa/capstone/raw/main/user.png"):
            st.markdown(prompt)

        try:
            # Classify the user input
            topic, sentiment, emotion = classify_sentiment_and_emotion(prompt)

            # Check if all classifications are unknown
            if topic == "Unknown" and sentiment == "Unknown" and emotion == "Unknown":
                st.session_state.conversation_stage = -1
            # Set conversation stage for support-only request
            elif "support" in prompt.lower() and "strategy" not in prompt.lower():
                st.session_state.conversation_stage = -2
            else:
                # Increment the conversation stage normally
                st.session_state.conversation_stage += 1

            # Generate the response
            assistant_response = generate_therapeutic_response(
                prompt, topic, sentiment, emotion, st.session_state.conversation_stage
            )

            with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
                st.markdown(assistant_response)

            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

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
