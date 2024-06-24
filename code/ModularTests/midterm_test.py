import chainlit as cl
import openai


# Set your API key and base URL
openai.api_key = "FAKE_API_KEY"
openai.api_base = "http://157.242.192.217/v1"  # Ensure this points to the correct API version endpoint

settings = {
    "model": "llama3",
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


    response = openai.ChatCompletion.create(
        model=settings["model"],
        messages=message_history,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        stream=True
    )

    async for part in response:
        if 'choices' in part and 'delta' in part['choices'][0] and 'content' in part['choices'][0]['delta']:
            token = part['choices'][0]['delta']['content']
            await msg.stream_token(token)

        # Append the assistant's last response to the history
    message_history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("message_history", message_history)

    # Update the message after streaming completion
    await msg.update()




