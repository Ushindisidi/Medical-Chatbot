# ingest.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings 
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import sys # Import sys to allow for graceful exit

# Load environment variables
load_dotenv()

# --- Configuration and Environment Variable Loading ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD_REGION = os.getenv("PINECONE_CLOUD_REGION") 
# HUGGINGFACEHUB_API_TOKEN is not directly used in ingest.py for embeddings, 
# but it's good practice to load it if other parts of your system might use it.
# However, HuggingFaceEmbeddings doesn't typically require this env var for pre-trained models.
# It's more for models that require authentication or are hosted privately.
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

# Check for essential environment variables
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found in .env file. Please check your .env configuration.")
    sys.exit(1) # Use sys.exit(1) for a non-zero exit code indicating an error
if not PINECONE_CLOUD_REGION:
    print("Error: PINECONE_CLOUD_REGION not found in .env file. Please check your .env configuration (e.g., 'us-east-1').")
    sys.exit(1)

PINECONE_INDEX_NAME = "medical-chatbot-index" 
DATA_PATH = "medical_data" 

# --- Initialize Pinecone client ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_CLOUD_REGION) 
    print("Pinecone client initialized successfully.")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    print("Please ensure your Pinecone API Key and Cloud Region (e.g., 'us-east-1') are correct in the .env file.")
    sys.exit(1)

# --- Load Embedding Model ---
print("Loading embedding model (all-MiniLM-L6-v2)... This may take a moment the first time.")
try:
    # dimension for all-MiniLM-L6-v2 is 384, consistent with your create_index call
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading HuggingFaceEmbeddings model: {e}")
    print("Please ensure you have 'sentence-transformers' installed correctly and an active internet connection.")
    sys.exit(1)

# --- Data Ingestion and Chunking Functions ---
def load_documents(data_path):
    """Loads PDF documents from a specified directory."""
    print(f"Loading documents from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist.")
        sys.exit(1)
    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            print(f"Warning: No PDF documents found in '{data_path}'. Please ensure the folder contains PDF files.")
            # We can continue here, but later ingestion might fail if no documents
        print(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please ensure your 'medical_data' folder exists and contains valid PDF files, and 'pypdf' is installed.")
        sys.exit(1)

def split_documents(documents):
    """Splits documents into smaller, overlapping text chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len,
        add_start_index=True, # Useful for debugging and traceability
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def ingest_data_to_pinecone(chunks, embeddings_model, index_name, pinecone_client, dimension=384, metric="cosine"):
    """
    Connects to or creates a Pinecone index and uploads document embeddings.
    """
    print(f"Checking for Pinecone index '{index_name}'...")
    
    try:
        # Check if the index already exists
        index_names = [index.name for index in pinecone_client.list_indexes()]
        if index_name not in index_names:
            print(f"Index '{index_name}' does not exist. Creating it now...")
            # Corrected: Use the full PINECONE_CLOUD_REGION directly for ServerlessSpec region
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension, 
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region=PINECONE_CLOUD_REGION) # Changed 'aws'.split('-')[0] to "aws"
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists. Connecting to it.")

        # Connect to the Pinecone vector store
        # This creates an empty vector store object connected to the Pinecone index
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings_model
        )

        # Upload in smaller batches to avoid hitting API limits or memory issues
        print("Uploading chunks to Pinecone in batches...")
        batch_size = 50 # You can adjust this based on your needs and Pinecone limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            print(f"Uploaded batch {i // batch_size + 1} of {len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)}")

        print("Data ingestion to Pinecone complete!")
        return vectorstore

    except Exception as e:
        print(f"Error during Pinecone ingestion: {e}")
        print("Please check your Pinecone setup, API key, region, and ensure your account has sufficient capacity.")
        sys.exit(1)

# --- Main execution ---
if __name__ == "__main__":
    print("Starting data ingestion process...")
    
    # Load documents from medical_data folder
    documents = load_documents(DATA_PATH)
    
    # Exit if no documents were loaded, as there's nothing to process
    if not documents:
        print("No documents to process. Exiting.")
        sys.exit(0) # Exit with success code if no documents were found but no error occurred

    # Split documents into chunks
    text_chunks = split_documents(documents)
    
    # Exit if no chunks were created
    if not text_chunks:
        print("No text chunks created. Exiting.")
        sys.exit(0)

    # Ingest data into Pinecone
    # The dimension for "all-MiniLM-L6-v2" is 384, which is correctly set.
    pinecone_vectorstore = ingest_data_to_pinecone(text_chunks, embeddings_model, PINECONE_INDEX_NAME, pc, dimension=384)
    
    print("Data ingestion script finished.")