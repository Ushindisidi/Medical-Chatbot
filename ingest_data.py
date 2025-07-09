import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings 
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD_REGION = os.getenv("PINECONE_CLOUD_REGION") 
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not all([PINECONE_API_KEY, PINECONE_CLOUD_REGION]):
    print("Error: Pinecone API Key or Cloud Region not found in .env file. Please check your .env configuration.")
    exit()

PINECONE_INDEX_NAME = "medical-chatbot-index" 
DATA_PATH = "medical_data" 

# 2. Initialize Pinecone client
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_CLOUD_REGION) 
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    print("Please ensure your Pinecone API Key and Cloud Region (e.g., us-east-1) are correct in the .env file.")
    exit()

print("Loading embedding model... This may take a moment the first time.")
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading HuggingFaceEmbeddings model: {e}")
    print("Please ensure you have 'sentence-transformers' installed correctly.")
    exit()


#  Data Ingestion and Chunking
def load_documents(data_path):
    print(f"Loading documents from {data_path}...")
    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please ensure your 'medical_data' folder exists and contains valid PDF files.")
        exit()

def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters per chunk
        chunk_overlap=200,    # Overlap 
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

# connect to Pinecone index and upload embeddings
def ingest_data_to_pinecone(chunks, embeddings_model, index_name, pinecone_client, dimension=384, metric="cosine"):
    print(f"Checking for Pinecone index '{index_name}'...")
    
    try:
        # Check if the index already exists
        index_names = [index.name for index in pinecone_client.list_indexes()]
        if index_name not in index_names:
            print(f"Index '{index_name}' does not exist. Creating it now...")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension, 
                metric=metric,       
                spec=ServerlessSpec(cloud="aws".split('-')[0], region=PINECONE_CLOUD_REGION)
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists. Connecting to it.")

        # Connect to the Pinecone vector store
        # Create an empty vector store connected to the Pinecone index
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings_model
        )

        # Upload in smaller batches
        print("Uploading chunks to Pinecone in batches...")
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            print(f"Uploaded batch {i // batch_size + 1} of {len(chunks) // batch_size + 1}")

        print("Data ingestion to Pinecone complete!")
        return vectorstore

    except Exception as e:
        print(f"Error during Pinecone ingestion: {e}")
        print("Please check your Pinecone setup, API key, and region.")
        exit()

# --- Main execution ---
if __name__ == "__main__":
    print("Starting data ingestion process...")
    
    # Load documents from medical_data folder
    documents = load_documents(DATA_PATH)
    
    # Split documents into chunks
    text_chunks = split_documents(documents)
    
    # 3. Ingest data into Pinecone
    pinecone_vectorstore = ingest_data_to_pinecone(text_chunks, embeddings_model, PINECONE_INDEX_NAME, pc, dimension=384)
    
    print("Data ingestion script finished.")
