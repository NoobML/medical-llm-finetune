from eda.eda import load_data, dataframe_analysis
from data.data_processing import get_data
import os
from datasets import DatasetDict
import pandas as pd

def test_plot_saving():
    assert os.path.exists("plots/eda_plots/label_distribution.png")
    assert os.path.exists("plots/eda_plots/Input_length_distribution.png")

def test_eda():
    dataset , dataframe = get_data()
    assert dataset is not None
    assert dataframe is not None

def test_answer_letter_column():
    dataset, dataframe = load_data()
    assert isinstance(dataset, DatasetDict)
    assert isinstance(dataframe, pd.DataFrame)
    dataframe_analysis(dataframe)
    assert 'answer_letter' in dataframe.columns


