import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re

# Load the Hugging Face model and tokenizer
model_name = "microsoft/DialoGPT-medium"
chatbot = pipeline("text-generation", model=model_name)

# Initialize conversation history
conversation_history = []

# Function to generate chatbot response
def chatbot_response(prompt):
    global conversation_history

    #write your code below
    

    # Display conversation history (loops and data structures)
    history = "\n".join(conversation_history[-6:])  # Show last 3 interactions
    
    return history

# Create a Gradio interface below

