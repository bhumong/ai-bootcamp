import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import random
import logging
import operator

# Load the Hugging Face model and tokenizer
model_name = "microsoft/DialoGPT-medium"

# Initialize conversation history
conversation_history = []
chatbot = None
tokenizer = None
model = None
chat_history_ids = []

# Function to generate chatbot response
def init_model():
    global chatbot
    global tokenizer
    global model
    
    if chatbot is None:
        chatbot = pipeline("text-generation", model=model_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
def safe_eval(expr):
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }
    
    parts = expr.split()
    if len(parts) == 3:
        left, op, right = parts
        left, right = int(left), int(right)
        if op in ops:
            return ops[op](left, right)
    raise ValueError("Invalid expression")

def chatbot_response(prompt, history = []):
    global conversation_history
    global chatbot
    global tokenizer
    global model
    global chat_history_ids

    init_model()
    prompt = f"{prompt}"
    response_bot = ""
    if prompt == "hello":
        response_bot = "Hi there! How can I help you today?"
    elif prompt == "bye":
        response_bot = "Goodbye! Have a great day!"
    elif "calculate " in prompt:
        try:
            calculate = prompt.split("calculate ")
            if len(calculate) != 2:
                raise ValueError("Invalid promt")
            calculate = f"{calculate[1]}"
            result = safe_eval(calculate)
            response_bot = f"The result is {result}"
        except Exception as e:
            print(e)
            response_bot = "Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'."
    else:
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(history) > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response_bot = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if (len(history) == 0):
        conversation_history.append(prompt)
        conversation_history.append(response_bot)
    else:
        conversation_history = history

    return response_bot

if __name__ == '__main__':
    # Create a Gradio interface below
    demo = gr.ChatInterface(
        fn=chatbot_response,
        type="messages"
    )

    demo.launch()