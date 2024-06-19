import streamlit as st
import requests

# Define the paths to your image files
logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"

# Display the logo at the top of the app
st.image(logo_path, width=300)

# Add a title and description
st.title("CHLA Chatbot Prototype")
st.write("Welcome to the Children's Hospital Los Angeles Chatbot Prototype.")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display the similarity threshold slider above the conversation
similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, key="slider")

# Function to send a query to the backend
def send_query(user_prompt):
    try:
        response = requests.post(
            "http://10.3.8.195:8000/query/",
            json={"user_prompt": user_prompt, "similarity_threshold": st.session_state.slider}
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        # Return a static message for any connection-related errors
        return {"generated_response": "Apologies for the inconvenience. System still in progress."}

# Display the conversation history
st.write("### Conversation")
for chat in st.session_state.conversation:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# Temporary variable to handle user input
temp_input = st.text_input("You:", key="user_input_temp")

if st.button("Send"):
    if temp_input:
        st.session_state.conversation.append({"user": temp_input, "bot": "..."})  # Placeholder for bot response
        result = send_query(temp_input)
        if result:
            bot_response = result['generated_response']
            st.session_state.conversation[-1]['bot'] = bot_response
        else:
            st.session_state.conversation[-1]['bot'] = "Error: Could not retrieve results from the backend."
        # Clear the temporary input field
        st.session_state.user_input_temp = ""  # Clear the input field
        st.experimental_rerun()  # Refresh the app to display the new conversation

# Display the icon at the bottom
st.image(icon_path, width=100)
