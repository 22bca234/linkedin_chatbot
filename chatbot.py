import streamlit as st
from transformers.pipelines.conversational import Conversation
from transformers import pipeline

# Set Streamlit page title and layout
st.set_page_config(page_title="Chatbot", layout="centered")

# Load the conversational AI model
@st.cache_resource
def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

chatbot = load_chatbot()

st.title("🤖 ProConnect AI Chatbot")
st.write("Ask me anything!")

# Initialize session state for conversation history and input counter
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# Create a placeholder container for the text input
input_placeholder = st.empty()

# Use a dynamic key for the text input widget
input_key = f"user_input_{st.session_state.input_counter}"
user_input = input_placeholder.text_input("You:", key=input_key)

if st.button("Send"):
    if user_input:
        # Process the user input
        conversation = Conversation(user_input)
        chatbot(conversation)
        response = conversation.generated_responses[-1]
        
        # Update chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        
        # Increment the input counter so that a new text input widget gets a new key
        st.session_state.input_counter += 1
        
        # Clear and re-create the text input widget with a new unique key (this clears the visible text)
        input_placeholder.empty()
        input_placeholder.text_input("You:", key=f"user_input_{st.session_state.input_counter}")

# Display chat history
for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}:** {message}")
