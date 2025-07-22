import os
import sys
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings # Or from langchain_huggingface.embeddings if you've installed it
import torch
from pinecone import Pinecone, ServerlessSpec # <--- UPDATED IMPORT: Import Pinecone class and ServerlessSpec

# Load environment variables FIRST
load_dotenv()

# --- Configuration and Environment Variable Loading ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # This is often like "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Embeddings Model Configuration ---
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Global Pinecone Client Instance (NEW WAY to initialize Pinecone) ---
# Initialize the Pinecone client globally once.
# Note: For Serverless indexes, the 'environment' might not be strictly necessary
# in the Pinecone() constructor if your API key is region-specific, but it's good practice
# to keep it consistent with your Pinecone setup.
try:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    
    # We create a global Pinecone client instance
    # The 'environment' parameter might be inferred or not needed depending on your Pinecone account type (serverless vs. pod-based)
    # If you are using a serverless index, the region is typically embedded in the API key.
    # For pod-based, you might still need the environment.
    # Check your Pinecone dashboard to confirm your setup.
    global_pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized globally.")

    # Optional: Verify the index exists. This part is for robust error checking.
    # This check is only relevant if you want to ensure the index is ready at startup.
    # If your index is created on demand or by another process, you might remove this.
    if PINECONE_INDEX_NAME not in global_pinecone_client.list_indexes().names():
        print(f"Error: Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
        # If you need to create it programmatically, you would do it here:
        # global_pinecone_client.create_index(
        #     name=PINECONE_INDEX_NAME,
        #     dimension=384, # all-MiniLM-L6-v2 uses 384 dimensions
        #     metric='cosine', # Or 'euclidean' based on your index creation
        #     spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT) # Use ServerlessSpec for serverless
        # )
        # print(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
        sys.exit(1) # Exit if index doesn't exist and we're not creating it
    
except ValueError as ve:
    print(f"Configuration Error: {ve}")
    print("Please ensure PINECONE_API_KEY is set in your .env file.")
    sys.exit(1)
except Exception as e:
    print(f"Error initializing Pinecone client or checking index: {e}")
    print("Please check your PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME.")
    sys.exit(1)


# --- LLM Loading ---
def load_llm():
    """Initializes and returns the Google Gemini LLM."""
    print("Loading Google Gemini LLM (gemini-pro)...")
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file or system environment.")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
        print("Gemini LLM loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading Google Gemini LLM: {e}")
        print("Please ensure GOOGLE_API_KEY is valid and the Gemini API is enabled for your Google Cloud project.")
        sys.exit(1)

# --- Embeddings Loading ---
def load_embeddings():
    """Initializes and returns the HuggingFace embeddings model."""
    print(f"Loading HuggingFace embeddings model: {EMBEDDINGS_MODEL_NAME}...")
    try:
        # Explicitly set device to 'cpu' to mitigate "meta tensor" errors
        device = "cpu" 
        print(f"Loading embeddings model on device: {device}")

        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL_NAME,
            model_kwargs={'device': device}
        )
        print("Embeddings model loaded successfully.")
        return embeddings_model
    except Exception as e:
        print(f"Error loading HuggingFace embeddings model ({EMBEDDINGS_MODEL_NAME}): {e}")
        print("This often indicates a PyTorch device or memory issue. Forcing CPU usage (device='cpu') is recommended.")
        sys.exit(1)

# --- Pinecone Vector Store Loading ---
def load_vectorstore():
    """Initializes and returns the Pinecone vector store."""
    print("Loading Pinecone vector store...")
    
    # We no longer need to pass api_key and environment to PineconeVectorStore
    # because the global Pinecone client instance handles the connection.
    try:
        embeddings = load_embeddings() # This loads the embeddings model
        
        # PineconeVectorStore now directly uses the global Pinecone client and index name
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
            # No need for api_key or environment here anymore
        )
        print("Pinecone vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error loading Pinecone vector store: {e}")
        print("Ensure the Pinecone index exists and the embeddings loaded correctly.")
        sys.exit(1)

# --- Create RetrievalQA Chain ---
def create_qa_chain(llm, vectorstore):
    """Creates and returns the RetrievalQA chain for question answering."""
    print("Creating RetrievalQA chain...")
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful and knowledgeable medical assistant. Use the provided context to answer the user's question accurately and thoroughly.
If you are unsure or if the answer is not in the context, respond politely with "I’m not sure about that. Please consult a medical professional." Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )
    print("RetrievalQA chain created.")
    return qa_chain

# --- Example Usage (Optional, for testing) ---
if __name__ == "__main__":
    print("Testing rag_utils functions...")
    
    # Load LLM
    llm = load_llm()
    
    # Load Vectorstore (which internally loads embeddings)
    vectorstore = load_vectorstore()
    
    # Create QA Chain
    qa_chain = create_qa_chain(llm, vectorstore)

    # Example query
    query = "What are the common symptoms of malaria?"
    print(f"\nAsking question: '{query}'")
    try:
        result = qa_chain.invoke({"query": query})
        print("\n--- Answer ---")
        print(result.get("result"))
        print("\n--- Sources ---")
        for doc in result.get("source_documents", []):
            print(f"- {doc.metadata.get('source', 'Unknown Source')} (page {doc.metadata.get('page', 'N/A')})")
    except Exception as e:
        print(f"Error during QA chain invocation: {e}")