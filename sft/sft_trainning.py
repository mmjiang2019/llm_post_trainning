import os
# 注意os.environ得在import huggingface库相关语句之前执行
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from llm_post_trainning.helper import (
    load_model_and_tokenizer,
    display_dataset,
    test_model_with_questions,
)

USE_GPU = False

model_name = "HuggingFaceTB/SmolLM2-135M"
model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)

train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
if not USE_GPU:
    train_dataset=train_dataset.select(range(100))

display_dataset(train_dataset)

# SFTTrainer config 
sft_config = SFTConfig(
    learning_rate=8e-5, # Learning rate for training. 
    num_train_epochs=1, #  Set the number of epochs to train the model.
    per_device_train_batch_size=1, # Batch size for each device (e.g., GPU) during training. 
    gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.
    gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
    logging_steps=2,  # Frequency of logging training progress (log every 2 steps).

)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset, 
    processing_class=tokenizer,
)
sft_trainer.train()

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?"
]
if not USE_GPU: # move model to CPU when GPU isn’t requested
    sft_trainer.model.to("cpu")
test_model_with_questions(sft_trainer.model, tokenizer, questions, 
                          title="Base Model (After SFT) Output")