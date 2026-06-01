from data.data_processing import get_data
import os

def test_get_data():
    dataset, df = get_data()
    assert dataset is not None # dataset loaded correctly
    assert df is not None
    assert len(df) > 0
    #-------
    assert 'input' in df.columns
    assert 'instruction' in df.columns
    assert  'output' in df.columns
    assert len(df) == 10178

def test_get_csv():
    get_data()
    assert os.path.exists("data/Medical_data.csv")


if __name__ == "__main__":
    test_get_data()
    test_get_csv()