# Medical Chatbot

## Project Overview

This project develops an intelligent medical chatbot designed to provide users with accurate and helpful information based on a curated knowledge base. Utilizing Retrieval-Augmented Generation (RAG) principles, the chatbot combines the power of large language models with external, domain-specific data stored in a vector database to deliver precise and context-aware medical answers.

**Disclaimer:** This chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any medical concerns.

## Features

* **Intelligent Q&A:** Answers medical questions using information from a reliable knowledge base.
* **Contextual Understanding:** Leverages a vector database (Pinecone) to retrieve relevant medical documents.
* **Large Language Model (LLM) Integration:** Uses Google Gemini for natural language understanding and generation.
* **Embeddings for Semantic Search:** Employs `all-MiniLM-L6-v2` HuggingFace embeddings for efficient semantic search in the vector store.
* **Streamlit User Interface:** Provides an intuitive and easy-to-use web interface for interaction.
* **Dockerization (Planned/Future):** Easy deployment and environment consistency using Docker containers.

## Technology Stack

* **Python:** Primary programming language.
* **LangChain:** Framework for building LLM applications.
* **Google Gemini API:** For the Large Language Model (LLM).
* **Pinecone:** Vector database for efficient storage and retrieval of medical knowledge embeddings.
* **HuggingFace `sentence-transformers`:** For generating text embeddings (`all-MiniLM-L6-v2`).
* **Streamlit:** For creating the interactive web application.
* **python-dotenv:** For managing environment variables.
* **Torch:** Underlying library for embeddings model operations.

## Setup and Installation

Follow these steps to set up and run the medical chatbot locally.

### 1. Clone the Repository

```bash
git clone <https://github.com/Ushindisidi/Medical-Chatbot.git>
cd <Medical-chatbot>
```

# Medical Chatbot

## Project Overview

This project develops an intelligent medical chatbot designed to provide users with accurate and helpful information based on a curated knowledge base. Utilizing Retrieval-Augmented Generation (RAG) principles, the chatbot combines the power of large language models with external, domain-specific data stored in a vector database to deliver precise and context-aware medical answers.

**Disclaimer:** This chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any medical concerns.

## Features

* **Intelligent Q&A:** Answers medical questions using information from a reliable knowledge base.
* **Contextual Understanding:** Leverages a vector database (Pinecone) to retrieve relevant medical documents.
* **Large Language Model (LLM) Integration:** Uses Google Gemini for natural language understanding and generation.
* **Embeddings for Semantic Search:** Employs `all-MiniLM-L6-v2` HuggingFace embeddings for efficient semantic search in the vector store.
* **Streamlit User Interface:** Provides an intuitive and easy-to-use web interface for interaction.
* **Dockerization (Planned/Future):** Easy deployment and environment consistency using Docker containers.

## Technology Stack

* **Python:** Primary programming language.
* **LangChain:** Framework for building LLM applications.
* **Google Gemini API:** For the Large Language Model (LLM).
* **Pinecone:** Vector database for efficient storage and retrieval of medical knowledge embeddings.
* **HuggingFace `sentence-transformers`:** For generating text embeddings (`all-MiniLM-L6-v2`).
* **Streamlit:** For creating the interactive web application.
* **python-dotenv:** For managing environment variables.
* **Torch:** Underlying library for embeddings model operations.

## Setup and Installation

Follow these steps to set up and run the medical chatbot locally.

 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```
2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv medical_chatbot_env
# On Windows
medical_chatbot_env\Scripts\activate
# On macOS/Linux
source medical_chatbot_env/bin/activate
```
3. Install Dependencies
Install all required Python packages:

```Bash
pip install -r requirements.txt
```
Markdown

# Medical Chatbot

## Project Overview

This project develops an intelligent medical chatbot designed to provide users with accurate and helpful information based on a curated knowledge base. Utilizing Retrieval-Augmented Generation (RAG) principles, the chatbot combines the power of large language models with external, domain-specific data stored in a vector database to deliver precise and context-aware medical answers.

**Disclaimer:** This chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any medical concerns.

## Features

* **Intelligent Q&A:** Answers medical questions using information from a reliable knowledge base.
* **Contextual Understanding:** Leverages a vector database (Pinecone) to retrieve relevant medical documents.
* **Large Language Model (LLM) Integration:** Uses Google Gemini for natural language understanding and generation.
* **Embeddings for Semantic Search:** Employs `all-MiniLM-L6-v2` HuggingFace embeddings for efficient semantic search in the vector store.
* **Streamlit User Interface:** Provides an intuitive and easy-to-use web interface for interaction.
* **Dockerization (Planned/Future):** Easy deployment and environment consistency using Docker containers.

## Technology Stack

* **Python:** Primary programming language.
* **LangChain:** Framework for building LLM applications.
* **Google Gemini API:** For the Large Language Model (LLM).
* **Pinecone:** Vector database for efficient storage and retrieval of medical knowledge embeddings.
* **HuggingFace `sentence-transformers`:** For generating text embeddings (`all-MiniLM-L6-v2`).
* **Streamlit:** For creating the interactive web application.
* **python-dotenv:** For managing environment variables.
* **Torch:** Underlying library for embeddings model operations.

## Setup and Installation

Follow these steps to set up and run the medical chatbot locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```
2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```Bash
python -m venv medical_chatbot_env
# On Windows
medical_chatbot_env\Scripts\activate
# On macOS/Linux
source medical_chatbot_env/bin/activate
```
3. Install Dependencies
Install all required Python packages:

```Bash

pip install -r requirements.txt
```
4. Configure Environment Variables
Create a .env file in the root directory of your project and populate it with your API keys and configuration details:

```bash
# Pinecone API Keys
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT" # e.g., "us-east-1"
PINECONE_INDEX_NAME="YOUR_PINECONE_INDEX_NAME"

# Google Gemini API Key
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
```
5.To run the Streamlit chatbot application:

```Bash

streamlit run app.py
```