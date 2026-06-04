import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from src.utils import prompt_formatting, get_merged_data
import torch
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import yaml
from data.data_processing import get_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_config():
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()


def load_tokenizer():
    return AutoTokenizer.from_pretrained(CONFIG['model']['name'])

def tokenize_data(data, tokenizer):
    tokenized_data = []
    for i in range(len(data)):
        row = prompt_formatting(data.iloc[i])
        tokenized_text = tokenizer(row)
        tokenized_data.append(tokenized_text)
    return tokenized_data

def detokenize_data(data, tokenizer):
    detokenized_text = []
    for i in range(len(data)):
        text = tokenizer.decode(data.iloc[i])
        detokenized_text.append(text)
    return detokenized_text

def get_bnb_config():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    return bnb

def load_model(bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
       CONFIG['model']['name'] ,
        quantization_config=bnb_config,
        device_map='auto'
    )
    return model

def get_peft_config():
    lora_config = LoraConfig(
        r=CONFIG['lora']['r'],
        lora_alpha=CONFIG['lora']['alpha'],
        lora_dropout=CONFIG['lora']['dropout'],
        task_type='CAUSAL_LM',
        inference_mode=False,
        target_modules=['o_proj', 'qkv_proj']
    )
    return lora_config

def apply_lora(model, lora_config):
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def train(model, dataset):
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=CONFIG['training']['epochs'],
        learning_rate=CONFIG['training']['learning_rate'],
        per_device_train_batch_size=CONFIG['training']['batch_size'],
        logging_steps=CONFIG['training']['logging_steps'],
        max_length=CONFIG['training']['max_length']
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )
    trainer.train()
    return trainer

def prepare_dataset(dataframe):
    merged_data = get_merged_data(dataframe)
    return Dataset.from_dict({'text': merged_data})

def save_model(model, tokenizer):
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    _ , dataframe = get_data()
    dataset = prepare_dataset(dataframe)
    tokenizer = load_tokenizer()
    bnb_config = get_bnb_config()
    model = load_model(bnb_config)
    peft_config = get_peft_config()
    lora_model = apply_lora(model, peft_config)
    trainer = train(lora_model, dataset)
    save_model(trainer.model, tokenizer)
