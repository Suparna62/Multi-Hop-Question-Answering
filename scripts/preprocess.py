import pandas as pd
import json
from tqdm import tqdm
import os
from datasets import DatasetDict, Dataset

def preprocess_cc_tsv(input_file, train_output_file, dev_output_file):
    """
    Preprocesses the cc.tsv dataset, splits it, and saves for training/eval.
    """
    df = pd.read_csv(input_file, sep='\t')
    

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    dev_df = df.iloc[split_index:]

    def df_to_jsonl(df, output_file):
        decomposed_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing to {output_file}"):
            question = row['Question']
            sub_questions_raw = f"{row['Q1']}; {row['Q2']}"
            answer = row['A1']
            
            decomposed_data.append({
                "question": question,
                "sub_questions": sub_questions_raw,
                "answer": answer
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in decomposed_data:
                f.write(json.dumps(item) + '\n')
    
    df_to_jsonl(train_df, train_output_file)
    df_to_jsonl(dev_df, dev_output_file)

    print(f"Processed {len(train_df)} training examples and {len(dev_df)} dev examples.")
    print(f"Saved to {train_output_file} and {dev_output_file}")


if __name__ == "__main__":
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
        
    preprocess_cc_tsv('data/cc.tsv', 'data/processed/train_decomposed.jsonl', 'data/processed/dev_decomposed.jsonl')

