import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import load_config, prepare_dataset, get_peft_config, get_bnb_config
from data.data_processing import get_data
from datasets import DatasetDict, Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig



def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert 'model' in config.keys()
    assert 'lora' in config.keys()
    assert 'training' in config.keys()

def test_prepare_dataset():
    _ , dataframe = get_data()
    result = prepare_dataset(dataframe)
    assert isinstance(result, Dataset)
    assert 'text' in result.column_names

def test_lora_config():
    obj = get_peft_config()
    assert isinstance(obj, LoraConfig)

def test_bnb_config():
    obj = get_bnb_config()
    assert isinstance(obj, BitsAndBytesConfig)




