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

# Initialize session state for conversation history and input control
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Check if we need to clear the input field by setting a default empty value
default_input = "" if st.session_state.clear_input else None

# Create text input with the dynamic default value
user_input = st.text_input("You:", key="user_input", value=default_input)

if st.button("Send"):
    if user_input:
        conversation = Conversation(user_input)
        chatbot(conversation)
        response = conversation.generated_responses[-1]
        
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        
        # Set flag to clear input on next run
        st.session_state.clear_input = True
        # Optionally, you can trigger a rerun to update the widget:
        st.experimental_rerun()

# Display chat history
for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}:** {message}")
