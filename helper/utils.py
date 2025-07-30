# Warning control
import warnings
warnings.filterwarnings('ignore')

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from IPython.display import display

def generate_responses(model, tokenizer, user_message=None, system_message=None, full_message=None, 
                       max_new_tokens=100):
    if full_message:
        messages = full_message
    else:
        # Format chat using tokenizer's chat template
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # We assume the data are all single-turn conversation
        messages.append({"role": "user", "content": user_message})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Recommended to use vllm, sglang or TensorRT
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response

def test_model_with_questions(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer,
        questions: list[str], 
        system_message: str=None, 
        title: str="Model Output",
    ):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

def load_model_and_tokenizer(model_name, use_gpu = False):
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if use_gpu:
        model.to("cuda")
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def display_dataset(dataset: Dataset):
    # Visualize the dataset
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages'] if m['role']=='user')
        assistant_msg = next(m['content'] for m in example['messages'] if m['role']=='assistant')

        rows.append({
            'User Message': user_msg,
            'Assistant Response': assistant_msg
        })

    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
    display(df)