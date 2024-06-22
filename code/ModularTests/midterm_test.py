import streamlit as st
import requests

logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"

# send a query to the backend
def send_query(user_prompt):
    try:
        response = requests.post(
            "http://10.3.8.195:8000/query/",
            json={"user_prompt": user_prompt, "similarity_threshold": st.session_state.slider}
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException:
        # Return a static message for any connection-related errors
        return {"generated_response": "Apologies for the inconvenience. System still in progress. Come back soon!"}

# boot chat interface
def boot():
    # display logo
    st.image(logo_path, width=300)

    # title and description
    st.title("CHLA Chatbot Prototype")
    st.write("Welcome to the Children's Hospital Los Angeles Chatbot Prototype. Ask a question regarding CHLA policy!")

    # initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # similarity threshold for testing
    st.session_state.slider = st.slider("Similarity Threshold (For Testing Purposes)", 0.0, 1.0, 0.7)

    # display the conversation history
    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    # user input for query
    if query := st.chat_input("Type your message..."):
        st.session_state.messages.append([query, ""])  # Placeholder for bot response
        st.chat_message("human").write(query)
        result = send_query(query)
        bot_response = result['generated_response']
        st.session_state.messages[-1][1] = bot_response
        st.chat_message("ai").write(bot_response)

if __name__ == "__main__":
    boot()

# Display the icon at the bottom
st.image(icon_path, width=100)
