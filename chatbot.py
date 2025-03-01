import streamlit as st
from transformers.pipelines.conversational import Conversation
from transformers import pipeline

st.set_page_config(page_title="Chatbot", layout="centered")

@st.cache_resource
def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

chatbot = load_chatbot()

st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create a placeholder container for the input field
input_placeholder = st.empty()

# Use a text_input inside the container; its default value can be empty
user_input = input_placeholder.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        conversation = Conversation(user_input)
        chatbot(conversation)
        response = conversation.generated_responses[-1]
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        # Clear the input by recreating the container with an empty text_input
        input_placeholder.text_input("You:", key="user_input", value="")

for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}:** {message}")
