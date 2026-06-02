import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing import get_data
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots", "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    dataset, dataframe = get_data()
    return dataset, dataframe

def dataset_analysis(dataset):

    print("=" * 30)
    print("Exploring the Original Dataset")
    print(f"Type of Dataset: \n{type(dataset)}\n --- \n")
    print(f"How many splits does the dataset contains: \n{dataset}\n --- \n")
    print(f"looking at the first row of the dataset:\n {dataset['train'][0]}")

def dataframe_analysis(dataframe):

    print("=" * 30)
    print("Converted Dataset into Pandas DataFrame")
    print(f"Dataframe Columns types: {dataframe.dtypes}")
    print(f"Missing values in dataset: \n{dataframe.isnull().sum()}")

    print("=" * 30)
    print("Checking Distribution of the Letters: (A/B/C/D/E)")
    dataframe['answer_letter'] = dataframe['output'].str[0]
    print(f"Value counts in numeric: \n{dataframe['answer_letter'].value_counts()} \n ------ \n")
    print("Visualizing it through the chart:")
    counts = dataframe['answer_letter'].value_counts()
    counts.plot(kind='bar', color='purple')
    plt.title("Answer Letter Distribution")
    plt.xlabel("Answer Letter")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "label_distribution.png"))
    plt.show()
    plt.clf()

    print("=" * 30)
    print("Checking for duplicate valeues")
    duplicate_values = dataframe.duplicated().sum()
    print(f"Duplicate value found:\n {duplicate_values}")

def input_length_analysis(dataframe):
    print("=" * 30)
    print("Input Length Analysis\n\n")
    input_length = dataframe['input'].str.len()
    print(f"INPUT LENGTH:\n{input_length.describe()}")
    input_length.plot(kind='hist', color='gray')
    plt.title("Input Length Distribution")
    plt.xlabel("Input Length")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Input_length_distribution.png"))
    plt.show()


def prompt_formatting(dataframe):
    print("=" * 30)
    print(f"Prompt Formatting")
    print(f"First 3 rows are: ")
    print("\n")
    for i in range(0, 3):
        print(f"Row: {i}:")
        row = dataframe.iloc[i]
        prompt = f"### Instruction:\n\n{row['instruction']}\n\n### Input:\n\n{row['input']}\n\n### Output:\n\n{row['output']}"
        print(prompt)
        print("\n\n")

def run_eda():
    dataset, dataframe = load_data()
    dataset_analysis(dataset)
    dataframe_analysis(dataframe)
    input_length_analysis(dataframe)
    prompt_formatting(dataframe)


if __name__ == "__main__":
    run_eda()
