import streamlit as st
import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import login
import openai

huggingface_api_key = st.secrets["huggingface"]["api_key"]
openai_api_key = st.secrets["openai"]["api_key"]

login(huggingface_api_key)

openai.api_key = openai_api_key

with open("label_mappings.json", "r") as f:
    label_mappings = json.load(f)

topic_labels = label_mappings["topic_labels"]
sentiment_labels = label_mappings["sentiment_labels"]
emotion_labels = label_mappings["emotion_labels"]

topic_model = RobertaForSequenceClassification.from_pretrained("isabelaal/roberta_topic_classifier")
sentiment_model = RobertaForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
emotion_model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

topic_model.eval()
sentiment_model.eval()
emotion_model.eval()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in [topic_model, sentiment_model, emotion_model]:
    model.to(device)

def classify_sentiment_and_emotion(user_input):
    """
    Classifies the input for topic, sentiment, and emotion.
    Returns identified values with basic confidence checks.
    """
    try:
        # Tokenize input for models
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        with torch.no_grad():
            # Predict topic
            topic_logits = topic_model(**inputs).logits
            topic_probs = torch.softmax(topic_logits, dim=-1).cpu().numpy()
            topic_idx = torch.argmax(topic_logits, dim=-1).item()
            topic_confidence = max(topic_probs[0])
            topic = topic_labels.get(str(topic_idx), "Unknown") if topic_confidence > 0.5 else "Unknown"

            # Predict sentiment
            sentiment_logits = sentiment_model(**inputs).logits
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1).cpu().numpy()
            sentiment_idx = torch.argmax(sentiment_logits, dim=-1).item()
            sentiment_confidence = max(sentiment_probs[0])
            sentiment = sentiment_labels.get(str(sentiment_idx), "Unknown") if sentiment_confidence > 0.5 else "Unknown"

            # Predict emotion
            emotion_logits = emotion_model(**inputs).logits
            emotion_probs = torch.softmax(emotion_logits, dim=-1).cpu().numpy()
            emotion_idx = torch.argmax(emotion_logits, dim=-1).item()
            emotion_confidence = max(emotion_probs[0])
            emotion = emotion_labels.get(str(emotion_idx), "Unknown") if emotion_confidence > 0.5 else "Unknown"

        # Debugging: Write classifications
        st.sidebar.write(
            f"Debug → Topic: {topic} (Confidence: {topic_confidence:.2f}), "
            f"Sentiment: {sentiment} (Confidence: {sentiment_confidence:.2f}), "
            f"Emotion: {emotion} (Confidence: {emotion_confidence:.2f})"
        )

        return topic, sentiment, emotion

    except Exception as e:
        # Handle errors
        st.sidebar.write(f"Classification Error: {e}")
        return "Unknown", "Unknown", "Unknown"


def generate_therapeutic_response(user_input, topic, sentiment, emotion, conversation_stage):
    # Base system message
    system_message = "You are a therapeutic assistant specializing in anxiety support."

    # Handle stage -1: Unknown classifications
    if conversation_stage == -1:
        user_prompt = (
            "The user's input is unclear. All classifications (topic, sentiment, and emotion) are unknown. "
            "Greet them warmly and encourage them to share more details about their thoughts or feelings. "
            "Avoid making assumptions and focus on open-ended questions to help them elaborate."
        )

    # Handle stage -2: Support-only requests
    elif conversation_stage == -2:
        user_prompt = (
            "The user has expressed that they want emotional support without suggestions or strategies. "
            "Respond empathetically, validate their feelings, and provide reassurance without offering advice."
        )

    # Handle gratitude for the bot (e.g., "Thanks for helping me")
    elif emotion == "gratitude" and topic == "Unknown":
        user_prompt = (
            "The user has expressed gratitude, possibly towards the assistant. "
            "Acknowledge their courage and progress in seeking support and addressing their concerns. "
            "Encourage them to continue taking steps forward, and ask if there's anything else they'd like to discuss or explore further."
        )

    # Handle stage 0: Beginning the conversation
    elif conversation_stage == 0:
        user_prompt = (
            "The user is starting the conversation. Greet them warmly, thank them for reaching out, and encourage them to share more about their feelings or thoughts."
        )

    # Handle stage 1: Exploring the user's feelings
    elif conversation_stage == 1:
        if topic == "Unknown":
            if sentiment != "Unknown":
                if emotion == "Unknown" or emotion == "Neutral":
                    user_prompt = (
                        "The user's input is unclear. Don't comment on their sentiment."
                        "Encourage them to share their thoughts, experiences, or any particular situations they’ve been facing."
                    )
                else:
                    user_prompt = (
                        f"The user feels {sentiment} and experiences {emotion}, but the specific topic is unclear. "
                        "Ask open-ended questions to explore their thoughts or any specific situations they’d like to discuss."
                    )
            elif emotion != "Unknown":
                user_prompt = (
                    f"The user experiences {emotion}, but their sentiment and the specific topic are unclear. "
                    "Validate their emotion and ask open-ended questions to help them share more about their current state or concerns."
                )
            else:
                user_prompt = (
                    "The topic, sentiment, and emotion are mostly unclear, but the user may need support. "
                    "Ask open-ended questions to explore their feelings or encourage them to share what’s been on their mind."
                )
        else:  # Topic is known
            if sentiment != "Unknown" and (emotion == "Unknown" or emotion == "Neutral"):
                user_prompt = (
                    f"The user feels {sentiment} about {topic}, but their emotions are unclear or neutral. "
                    "Encourage them to share more about what’s on their mind or how this topic is affecting them."
                )
            else:
                user_prompt = (
                    f"The user feels {sentiment} and experiences {emotion} about {topic}. "
                    "Acknowledge their context and ask questions to help them explore their feelings more deeply. "
                    "Encourage them to share specific aspects of the topic that might be contributing to their emotions."
                )

    
    # Handle stage 2: Identifying triggers and stressors
    elif conversation_stage == 2:
        user_prompt = (
            f"The user has shared feeling {sentiment} and experiencing {emotion} about {topic}. "
            "Ask them to identify specific aspects causing their anxiety or stress. Focus on understanding their main triggers."
        )

    # Handle stage 3: Introducing actionable techniques
    elif conversation_stage == 3:
        user_prompt = (
            f"The user is feeling {sentiment} and experiencing {emotion} about {topic}. "
            "Suggest actionable CBT techniques, such as thought challenging, reframing, or mindfulness exercises. "
            "Validate their feelings and encourage them to try these techniques."
        )

    # Handle stage 4: Providing detailed guidance
    elif conversation_stage == 4:
        user_prompt = (
            f"The user feels {sentiment} and experiences {emotion} about {topic}. "
            "Provide detailed guidance on using CBT techniques step by step, like thought records, exposure tasks, or journaling. "
            "Offer specific examples where possible."
        )

    # Handle stage 5: Wrapping up the conversation
    elif conversation_stage == 5:
        user_prompt = (
            f"The user has discussed their concerns about {topic}. Summarize their key concerns and progress. "
            "Reinforce their achievements, provide encouragement, and ask if there’s anything else they’d like to discuss. "
            "If they request more strategies, redirect to earlier stages."
        )

    # Handle unknown stages
    else:
        user_prompt = (
            "Summarize the key points discussed, express encouragement, and ask if there’s anything else the user would like to talk about. "
            "If they bring up a new topic, restart the conversation flow."
        )

    # Generate the response using OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"The user said: {user_input}. {user_prompt}"}
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

    if prompt := st.chat_input("Welcome! I'm here to help you manage anxiety and provide support. What's on your mind?"):
        if prompt.strip().lower() == "end session":
            st.session_state.clear()
            st.session_state.messages = []
            st.success("Session ended. Feel free to start a new conversation!")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="https://github.com/iisabelaaa/capstone/raw/main/user.png"):
            st.markdown(prompt)

        # Classify user input
        topic, sentiment, emotion = classify_sentiment_and_emotion(prompt)

        # Adjust conversation stage
        if topic == "Unknown" and sentiment == "Unknown" and emotion == "Unknown":
            st.session_state.conversation_stage = -1
        elif st.session_state.conversation_stage == 0:
            st.session_state.conversation_stage = 1
        else:
            st.session_state.conversation_stage += 1

        # Generate response
        assistant_response = generate_therapeutic_response(
            prompt, topic, sentiment, emotion, st.session_state.conversation_stage
        )

        with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
            st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})


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

if __name__ == "__main__":
    main()
