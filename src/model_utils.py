import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

def get_model_and_tokenizer(model_name="prometheus-eval/prometheus-7b-v2.0" ):
    """
    Load the Prometheus model and tokenizer.
    
    Args:
        model_name (str): The specific Prometheus model version to load.
        
    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables.")
        
    print(f"Loading model: {model_name}...")
    
    # Placeholder for actual loading logic to prevent large downloads during setup
    # In a real run, you would uncomment the following:
    tokenizer = None
    model = None
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     token=hf_token,
    #     device_map="auto"
    # )
    
    
    return model, tokenizer
