import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to access them
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_CLOUD_REGION") 
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(f"Pinecone API Key loaded: {'Yes' if pinecone_key else 'No'}")
print(f"Pinecone Cloud Region loaded: {'Yes' if pinecone_region else 'No'}") 
print(f"Hugging Face Token loaded: {'Yes' if hf_token else 'No'}")