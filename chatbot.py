import streamlit as st
from transformers.pipelines.conversational import Conversation

# Set Streamlit page title and layout
st.set_page_config(page_title="Chatbot", layout="centered")

# Load the conversational AI model
@st.cache_resource
def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

chatbot = load_chatbot()

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

# Initialize session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        conversation = Conversation(user_input)
        chatbot(conversation)
        response = conversation.generated_responses[-1]
        
        # Store chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        
        # Clear input
        st.session_state.user_input = ""

# Display chat history
for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}:** {message}")
