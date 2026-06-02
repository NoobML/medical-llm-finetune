import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, load_from_disk
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "raw_dataset")
CSV_PATH = os.path.join(BASE_DIR, "Medical_data.csv")


def import_data():
    dataset = load_dataset('medalpaca/medical_meadow_medqa')
    return dataset

def convert_to_dataframe(dataset):
    return dataset['train'].to_pandas()

def save_data_as_csv(df):
    df.to_csv(CSV_PATH, index=False)

def get_data():

    if os.path.exists(DATASET_PATH) and os.path.exists(CSV_PATH):
        dataset = load_from_disk(DATASET_PATH)
        df = pd.read_csv(CSV_PATH)
    else:
        dataset = import_data()
        df = convert_to_dataframe(dataset)
        save_data_as_csv(df)
        dataset.save_to_disk(DATASET_PATH)

    return dataset, df