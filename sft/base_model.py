import os
# 注意os.environ得在import huggingface库相关语句之前执行
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llm_post_trainning.helper.utils import (
    load_model_and_tokenizer,
    test_model_with_questions,
)

USE_GPU = False

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?"
]

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", USE_GPU)

test_model_with_questions(model, tokenizer, questions, 
                          title="Base Model (Before SFT) Output")

del model, tokenizer