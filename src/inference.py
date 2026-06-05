import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'training')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import yaml

def load_config():
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def load_model_for_inference():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    # base model
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model']['name'],
        quantization_config=bnb_config,
        device_map='auto'
    )
    # adding LoRA adapters on top
    model = PeftModel.from_pretrained(base_model, RESULT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model']['name'])
    return model, tokenizer

def format_input(question):
    prompt = f"### Instruction:\n\nPlease answer with one of the option in the bracket\n\n### Input:\n\n{question}\n\n### Output:\n\n"
    return prompt

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=CONFIG['inference']['max_new_tokens'])
    return output

def decode_output(tokenizer, output):
    result = tokenizer.decode(output, skip_special_tokens=True)
    return result

def predict(question):
    model, tokenizer = load_model_for_inference()
    prompt = format_input(question)
    output = generate_answer(model, tokenizer, prompt)
    answer = decode_output(tokenizer, output[0])
    return answer


if __name__ == "__main__":
    question = input("Enter your medical question: ")
    answer = predict(question)
    print(answer)



