import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def prompt_formatting(row):
    prompt = f"### Instruction:\n\n{row['instruction']}\n\n### Input:\n\n{row['input']}\n\n### Output:\n\n{row['output']}"
    return prompt


if __name__ == "__main__":
    pass
