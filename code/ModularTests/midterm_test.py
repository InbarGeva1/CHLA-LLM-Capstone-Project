import chainlit as cl
from langchain.llms import OpenAI

# Set your API key and base URL

server_ip = "http://157.242.192.217/v1"  # Ensure this points to the correct API version endpoint

llm = OpenAI(
    api_key="fake_key",
    base_url=server_ip,
    model="llama3"
)


settings = {
    "temperature": 0.7,
    "max_tokens": 500,
}

@cl.on_chat_start
def start_chat():
    # Initialize message history
    cl.user_session.set("message_history", [{"role": "system", "content": "You are a helpful chatbot."}])

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the message history from the session
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    # Create an initial empty message to send back to the user
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Use streaming to handle partial responses if supported
        response = await llm(messages=message_history, temperature=settings["temperature"], max_tokens=settings["max_tokens"])

        async for token in response:
            await msg.stream_token(token)

        # Append the assistant's last response to the history
        message_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("message_history", message_history)

        # Update the message after streaming completion
        await msg.update()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()