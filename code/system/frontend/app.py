import streamlit as st
import requests


def main():
    st.title("CHLA Chatbot Prototype")

    # User query
    user_prompt = st.text_input("Enter your query:")

    # User input for similarity threshold
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6)

    if st.button("Search"):
        if user_prompt:
            # Make a request to the backend API
            response = requests.post(
                "http://10.3.8.195:8000/query/",
                json={"user_prompt": user_prompt, "similarity_threshold": similarity_threshold}
            )

            if response.status_code == 200:
                result = response.json()
                # Display relevant documents and the generated response
                st.subheader("Relevant Documents:")
                for text in result['relevant_texts']:
                    st.markdown(f"**Document:**\n{text}")
                    st.markdown("---")

                st.subheader("Generated Response:")
                st.markdown(result['generated_response'])
            else:
                st.error("Error: Could not retrieve results from the backend.")
        else:
            st.error("Please enter a query.")


if __name__ == "__main__":
    main()
