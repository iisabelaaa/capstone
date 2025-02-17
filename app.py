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

    try:

        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        with torch.no_grad():

            topic_logits = topic_model(**inputs).logits
            topic_probs = torch.softmax(topic_logits, dim=-1).cpu().numpy()
            topic_idx = torch.argmax(topic_logits, dim=-1).item()
            topic_confidence = max(topic_probs[0])
            topic = topic_labels.get(str(topic_idx), "Unknown") if topic_confidence > 0.5 else "Unknown"

            sentiment_logits = sentiment_model(**inputs).logits
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1).cpu().numpy()
            sentiment_idx = torch.argmax(sentiment_logits, dim=-1).item()
            sentiment_confidence = max(sentiment_probs[0])
            sentiment = sentiment_labels.get(str(sentiment_idx), "Unknown") if sentiment_confidence > 0.5 else "Unknown"

            emotion_logits = emotion_model(**inputs).logits
            emotion_probs = torch.softmax(emotion_logits, dim=-1).cpu().numpy()
            emotion_idx = torch.argmax(emotion_logits, dim=-1).item()
            emotion_confidence = max(emotion_probs[0])
            emotion = emotion_labels.get(str(emotion_idx), "Unknown") if emotion_confidence > 0.5 else "Unknown"

        return topic, sentiment, emotion

    except Exception as e:
        return "Unknown", "Unknown", "Unknown"

def is_greeting(user_input):
    if not isinstance(user_input, str):
        return False
    greetings = ["hello", "hi", "hey", "greetings", "what's up", "howdy", "sup"]
    return user_input.strip().lower() in greetings


def generate_therapeutic_response(user_input, topic, sentiment, emotion, conversation_stage, conversation_history):

    system_message = "You are a conversational and supportive therapeutic assistant specializing in anxiety support."

    # Stage 0: Initial Greeting
    if conversation_stage == 0:
        user_prompt = (
            "The user has started the conversation. Greet them warmly and invite them to share their thoughts or feelings. "
            "Focus on creating a safe and open environment."
        )

    # Stage 1: Exploring the User's Feelings
    elif conversation_stage == 1:
        user_prompt = (
            f"The user feels {sentiment} and is experiencing {emotion} about {topic}. "
            "Acknowledge their feelings empathetically and ask an open-ended question to encourage further sharing."
        )

    # Stage 2: Identifying Stressors or Triggers
    elif conversation_stage == 2:
        user_prompt = (
            f"The user has shared feeling {sentiment} and experiencing {emotion} about {topic}. "
            "Help them explore specific stressors or triggers that might be contributing to their feelings."
        )

    # Stage 3: Offering Practical Strategies
    elif conversation_stage == 3:
        user_prompt = (
            f"The user feels {sentiment} and experiences {emotion} about {topic}. "
            "Introduce one practical CBT technique, such as thought reframing, grounding exercises, or journaling. "
            "Keep the explanation simple and actionable."
        )

    # Stage 4: Providing Detailed Guidance
    elif conversation_stage == 4:
        user_prompt = (
            f"The user is feeling {sentiment} and experiencing {emotion} about {topic}. "
            "Provide detailed guidance on using CBT techniques or mindfulness exercises step by step. "
            "Offer specific examples where appropriate."
        )

    # Stage 5: Wrapping Up the Conversation
    elif conversation_stage == 5:
        user_prompt = (
            "The user has been discussing their concerns and feelings in this conversation. "
            "Summarize the key points shared so far, acknowledge their progress, and reinforce any strategies or insights "
            "discussed. Provide encouragement and ask if there's anything else they would like to explore before concluding."
        )

    # Handle unknown stages or context
    else:
        user_prompt = (
            "Respond warmly and summarize the conversation so far. "
            "Ask the user if they’d like to revisit any topics or explore new concerns."
        )

    MAX_TURNS = 20
    if len(conversation_history) > MAX_TURNS:
        conversation_history = conversation_history[-MAX_TURNS:]

    
    # Build the messages from conversation history with summarization if too long
    if len(conversation_history) > MAX_TURNS:
        conversation_history = conversation_history[-MAX_TURNS:] 

    history_summary = " ".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history]
    )


    # Build the messages from conversation history
    messages = [{"role": "system", "content": system_message}] + conversation_history + [
        {"role": "user", "content": f"The user said: {user_input}. {user_prompt}"}
    ]

    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:personal::AYvfotKM",
        messages=messages
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
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": "Welcome! I'm here to help you manage anxiety and provide support. What's on your mind?"
        })

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

    # Handle user input with validation
    prompt = st.chat_input("Type here...")
    if prompt is None or prompt.strip() == "":
        return

    # Check for session end command
    if prompt.strip().lower() == "end session":
        st.session_state.clear()
        st.session_state.messages = []
        st.session_state.conversation_stage = 0
        st.success("Session ended. Feel free to start a new conversation!")
        return

    # Check if the user input is a greeting
    if is_greeting(prompt):
        st.session_state.conversation_stage = 0
        assistant_response = (
            "Hello! It's great to see you here. Feel free to share how you're feeling or what's been on your mind. "
            "I'm here to listen and support you."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
            st.markdown(assistant_response)
        return

    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="https://github.com/iisabelaaa/capstone/raw/main/user.png"):
        st.markdown(prompt)

    try:
        # Classify user input
        topic, sentiment, emotion = classify_sentiment_and_emotion(prompt)

        # Update conversation stage
        if topic == "Unknown" and sentiment == "Unknown" and emotion == "Unknown":
            st.session_state.conversation_stage = -1 
        elif st.session_state.conversation_stage == 0:
            st.session_state.conversation_stage = 1 
        else:
            st.session_state.conversation_stage += 1 

        # Generate the response using conversation history
        assistant_response = generate_therapeutic_response(
            user_input=prompt,
            topic=topic,
            sentiment=sentiment,
            emotion=emotion,
            conversation_stage=st.session_state.conversation_stage,
            conversation_history=st.session_state.messages 
        )


        # Display and add assistant response
        with st.chat_message("assistant", avatar="https://github.com/iisabelaaa/capstone/raw/main/assistant.png"):
            st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    except Exception as e:

        st.error(f"An error occurred: {e}")


        assistant_response = "I encountered an issue while processing your input. Could you please try rephrasing or sharing more details?"
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
