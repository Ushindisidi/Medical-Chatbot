# test_hf_login.py
import os
from dotenv import load_dotenv
from huggingface_hub import login, InferenceClient
from huggingface_hub.utils import HfHubHTTPError # Import from utils for older versions
load_dotenv() # Load your .env file

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if hf_token:
    print(f"Attempting to log in with token of length: {len(hf_token)}")
    try:
        login(token=hf_token)
        print("huggingface_hub.login() successful!")

        # Try a quick inference client call to confirm
        print("\nAttempting inference with Zephyr...")
        client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
        response = client.text_generation("Hello, who are you?", max_new_tokens=20)
        print(f"Zephyr Inference successful: {response}")

    except HfHubHTTPError as e:
        print(f"Hugging Face Hub HTTP Error during login or inference: {e}")
        print("This typically means the token is invalid or has insufficient permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
else:
    print("HUGGINGFACEHUB_API_TOKEN environment variable not found. Check your .env file.")