# CHLA-LLM-Capstone-Project

This repository contains the code and resources for the Infection & Prevention Control (IPC) Chatbot developed for Children's Hospital Los Angeles (CHLA). This chatbot provides instant access to CHLA's IPC policy documentation and references CDC guidance using a retrieval-augmented generation (RAG) architecture. 

## Project Overview

### The Problem
The current process for locating and retrieving IPC policies at CHLA is manual and labor-intensive. Staff members must navigate multiple sources to find relevant information, leading to inefficiencies and delays in accessing important guidance.

### Our Solution
We developed a Large Language Model (LLM) chatbot that delivers domain-specific content in a conversational format. The chatbot is designed to:
- Query and retrieve CHLA's policy documents and CDC guidance.
- Provide human-like responses with dynamic content retrieval.
- Keep internal content private while allowing growth over time.
- Be "plug-and-play" and agnostic, supporting the use of different pre-trained LLMs without costly retraining.

## Tech Stack

- **Google Custom Search API**: For referencing CDC content.
- **Ollama Model Wrapper**: LLM hosting infrastructure.
- **Llama 3.1 8B Model**: The core language model used.
- **Python**: Programming language for backend development.
- **Streamlit**: Frontend for user interaction.
- **LangChain**: RAG (Retrieval-Augmented Generation) pipeline.
- **ChromaDB**: Vector database for optimized document search and retrieval.
- **Ubuntu 22.04**: The OS environment used for development and deployment.

## Folder Structure

- `cdc_vectorstore/`: Contains vectorized data and search capabilities for CDC content.
- `chla_vectorstore/`: Contains vectorized data and search capabilities for CHLA policy documents.
- `code/`: The core code for the chatbot, including system setup.
  - `system/`: System-level files for setup and configuration.
  - `.gitignore`: Git configuration to ignore unnecessary files.
  - `requirements.txt`: The Python dependencies required for running the project.
- `data/`: Includes resources such as images and documentation.
  - `README.md`: Additional project details.
  - `childrens-hospital-la-icon.jpg`: CHLA branding icon.
  - `childrens-hospital-la-logo.png`: CHLA branding logo.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/CHLA-LLM-Capstone-Project.git
    ```
2. Navigate to the directory:
    ```bash
    cd CHLA-LLM-Capstone-Project
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the chatbot interface:
    ```bash
    streamlit run app.py
    ```

## Contributors

- Diego Estuar
- Andrew Morris
- Lance Royston
- Inbar Geva

**Advisors:**
- Dr. Arin Brahma
- Dr. Michael Neely
