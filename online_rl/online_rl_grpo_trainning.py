import re
import torch

# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
# 注意os.environ得在import huggingface库相关语句之前执行
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import transformers
transformers.logging.set_verbosity_error()

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from trl import GRPOTrainer, GRPOConfig

from llm_post_trainning.helper.utils import (
    load_model_and_tokenizer,
    generate_responses,
)


def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]['content']).group(1) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, otherwise reward 0
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

def post_processing(example):
    match = re.search(r"####\s*(-?\d+)", example['answer'])
    example['ground_truth'] = match.group(1) if match else None
    example['prompt'] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['question']}
    ]

    return example

# Prepare for evaluation dataset for Math: GSM8K
USE_GPU = False
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step-by-step."
    "Always include the final numeric answer inside \\boxed{}."
)

sample_pred = [
    [
        {"role": "assistant", "content": r"...Calculating the answer. \boxed{72}"}
    ]
]

ground_truth = ["72"]

reward = reward_func(sample_pred, ground_truth)
print(f"Negative Sample Reward: {reward}")

# Load the evaluation dataset
data_num = 5
eval_dataset = load_dataset("openai/gsm8k", "main")["test"].select(range(data_num))
sample_df = eval_dataset.to_pandas()
display(sample_df)

eval_dataset = eval_dataset.map(post_processing).remove_columns(["question", "answer"])
sample_df = eval_dataset.select(range(5)).to_pandas()
display(sample_df)

# Load the model and evaluate
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", USE_GPU)

# Store predictions and ground truths
all_preds = []
all_labels = []

for example in tqdm(eval_dataset):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]
    # Run the model to generate an answer
    with torch.no_grad():
        response = generate_responses(model, tokenizer, 
                                      full_message = input_prompt) 
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(response)
    print("Ground truth: ", ground_truth)

# 3. Evaluate using reward_func
rewards = reward_func(all_preds, all_labels)

# 4. Report accuracy
accuracy = sum(rewards) / len(rewards)
print(f"Evaluation Accuracy: {accuracy:.2%}")
del model, tokenizer

# Loading the training dataset
dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]
 
# Apply to dataset
train_dataset = train_dataset.map(post_processing)
train_dataset = train_dataset.remove_columns(["question", "answer"])
if not USE_GPU:
    train_dataset = train_dataset.select(range(10))
print(train_dataset[0])

# GRPO Training
config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4, # Can set as high as 64 or 128
    num_train_epochs=1,
    learning_rate=5e-6,
    logging_steps=2,
    no_cuda= not USE_GPU     # keeps the whole run on CPU, incl. MPS
)

## If this block hangs or the kernel restarts during training, please skip loading the previous 0.5B model for evaluation
model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct", USE_GPU)
grpo_trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=reward_func,
    train_dataset=train_dataset
)
grpo_trainer.train()

fully_trained_qwen = True
if fully_trained_qwen:
    model, tokenizer = load_model_and_tokenizer("banghua/Qwen2.5-0.5B-GRPO", 
                                            USE_GPU)
else:
    model = grpo_trainer.model

# Store predictions and ground truths
all_preds = []
all_labels = []

for example in tqdm(eval_dataset):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]
    # Run the model to generate an answer
    with torch.no_grad():
        response = generate_responses(model, tokenizer, 
                                      full_message = input_prompt) 
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(response)
    print("Ground truth: ", ground_truth)

# 3. Evaluate using reward_func
rewards = reward_func(all_preds, all_labels)

# 4. Report accuracy
accuracy = sum(rewards) / len(rewards)
print(f"Evaluation Accuracy: {accuracy:.2%}")