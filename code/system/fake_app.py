import chainlit as cl
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define the external context
context = """
PURPOSE:

To reduce the risk of surgical site infections through the implementation of prevention bundles for patients undergoing identified high risk surgical procedures at Children’s Hospital Los Angeles (CHLA).

SCOPE:

This policy is applicable to Neurosurgery, Cardiothoracic, and Orthopedics, Operating Room (OR), 6 East (6E), Cardiothoracic Intensive Care Unit (CTICU), Cardiovascular Acute (CVA), Emergency Department (ED).

PROCEDURES:

A. For the surgical procedures identified as high risk by the Infection Control Committee’s annual risk assessment (i.e. Cardiothoracic, Ventricular Shunt, and Spinal Fusion surgeries), a bundle of prevention practices consisting of the following standard elements shall be followed:
a. Pre-operative education for patients and their families
b. Pre-operative bathing and skin prophylaxis protocol
c. Pre-operative antimicrobial administration protocol
d. Surgical site skin preparation protocol
e. Post-operative education with emphasis on reporting signs and symptoms of infection
B. Changes to bundles must be approved by the Infection Control Committee.
C. Patients who are admitted prior to the identified high-risk surgeries (See appendices) will receive pre-operative education and Chlorohexidine (CHG) bathing prior to surgery.
a. Chlorohexidine (CHG) wipes may be used for all patients outside of the NICCU. For NICCU patients, please refer to NICCU’s current recommendations and guidelines.

REFERENCES:
1. HICPAC Guidelines for the Prevention of Surgical Site Infection, 1999. ICHE Vol.20, No 4, page 247.
2. APIC text of Infection Control and Epidemiology. Chapter 37: Surgical Site Infection, 2018.
3. Centers for Disease Control and Prevention Guideline for the Prevention of Surgical Site Infection, 2017. JAMA, Vol. 152, No. 8, pages 784-791.
4. 2017 HICPAC-CDC Guideline for Prevention of Surgical Site Infection: What the infection preventionist needs to know. APIC, Prevention Strategist, 2017.

ATTACHMENTS:
1. IC – 229.1 Appendix A Cardiac SSI Bundle
2. IC – 229.2 Appendix B Neuro SSI Bundle
3. IC – 229.3 Appendix C Ortho Spine SSI Bundle
4. IC – 229.4 Appendix D High Risk Pre-Operative Bathing Parent Education (English)
5. IC – 229.5 Appendix E High Risk Pre-Operative Bathing Parent Education (Spanish)

POLICY OWNER:
Manager, Infection Prevention and Control
"""

# Define the updated prompt template
prompt_template = PromptTemplate.from_template("""
Context: {context}

User Question: {input_text}

Please provide a detailed and natural-sounding answer based on the context above. Maintain all medical terminology and ensure the response is clear and concise.

Answer:
""")

# Initialize the Ollama model
ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434")

# Create the LLMChain
chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

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
        # Combine context and user message
        combined_prompt = prompt_template.format(context=context, input_text=message.content)

        # Run the chain with the combined prompt
        response = chain.run({"context": context, "input_text": message.content})

        # Stream the response
        async for token in response:
            await msg.stream_token(token)

        # Append the assistant's last response to the history
        message_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("message_history", message_history)

        # Update the message after streaming completion
        await msg.update()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
