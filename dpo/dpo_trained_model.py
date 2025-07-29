# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
# 注意os.environ得在import huggingface库相关语句之前执行
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import transformers
transformers.logging.set_verbosity_error()

from llm_post_trainning.helper.utils import (
    load_model_and_tokenizer,
    test_model_with_questions,
)

USE_GPU = False

questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization."
]

model, tokenizer = load_model_and_tokenizer("banghua/Qwen2.5-0.5B-DPO",
                                            USE_GPU)

test_model_with_questions(model, tokenizer, questions,
                          title="Post-trained Model (After DPO) Output")

del model, tokenizer