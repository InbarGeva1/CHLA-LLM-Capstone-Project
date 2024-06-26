import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define the external context
context = """
CHLA DOCUMENTATION:

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

CDC DOCUMENTATION:
Findings  Before surgery, patients should shower or bathe (full body) with soap (antimicrobial or nonantimicrobial) or 
an antiseptic agent on at least the night before the operative day. Antimicrobial prophylaxis should be administered only 
when indicated based on published clinical practice guidelines and timed such that a bactericidal concentration of the agents is 
established in the serum and tissues when the incision is made. In cesarean section procedures, antimicrobial prophylaxis should be 
administered before skin incision. Skin preparation in the operating room should be performed using an alcohol-based agent unless contraindicated. 
For clean and clean-contaminated procedures, additional prophylactic antimicrobial agent doses should not be administered after the surgical incision 
is closed in the operating room, even in the presence of a drain. Topical antimicrobial agents should not be applied to the surgical incision. 
During surgery, glycemic control should be implemented using blood glucose target levels less than 200 mg/dL, and normothermia should be maintained in all patients. 
Increased fraction of inspired oxygen should be administered during surgery and after extubation in the immediate postoperative period for patients with normal pulmonary 
function undergoing general anesthesia with endotracheal intubation. Transfusion of blood products should not be withheld from surgical patients as a means to prevent SSI.
"""

# Define the updated prompt template
prompt_template = PromptTemplate.from_template("""
Documentation: {context}

User Question: {input_text}

Please provide a detailed and natural-sounding answer based on the documentation above. Provide separate paragraphs of summarization for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear and concise. Use bullet points and step-by-step instructions for clarity when applicable.
Only provide the summarizations using the following markdown format and begin by your response by saying:

**CHLA Recommendation:**
(newline)
summary based on chla context

**CDC Recommendation:**
(newline)
summary based on cdc context

Attach this link at the end of the chla paragraph: https://lmu.app.box.com/file/1562757601538
Attach this link at the end of the CDC paragraph: https://www.cdc.gov/infection-control/hcp/surgical-site-infection/index.html

Answer:
""")

# Initialize the Ollama model
ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.3)

# Create the LLMChain
chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"


# boot chat interface
def boot():
    # Display logo
    st.image(logo_path, width=300)

    # Title and description
    st.title("CHLA Chatbot Prototype")
    st.write("Welcome to the Children's Hospital Los Angeles Chatbot Prototype. Ask a question regarding CHLA policy!")

    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Similarity threshold for testing
    st.session_state.slider = st.slider("Similarity Threshold (For Testing Purposes)", 0.0, 1.0, 0.7)

    # Display the conversation history
    for message in st.session_state.messages:
        if message[0] == "human":
            st.chat_message("human").write(message[1])
        else:
            st.chat_message("ai").write(message[1])

    # User input for query
    if query := st.chat_input("Type your message..."):
        st.session_state.messages.append(["human", query])  # Placeholder for bot response
        st.chat_message("human").write(query)

        # Combine context and user message
        combined_prompt = prompt_template.format(context=context, input_text=query)

        try:
            response = chain.run({"context": context, "input_text": query})
            st.session_state.messages.append(["ai", response])
        except Exception as e:
            response = f"An error occurred: {str(e)}"
            st.session_state.messages.append(["ai", response])

        st.chat_message("ai").write(response)


if __name__ == "__main__":
    boot()

# Display the icon at the bottom
st.image(icon_path, width=100)



