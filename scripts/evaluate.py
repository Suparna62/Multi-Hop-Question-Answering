import yaml
import pandas as pd
from datasets import load_metric, Dataset
from tqdm import tqdm
import ast
from src.pipeline.main_pipeline import load_pipeline_components, search_web, combine_results
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_system():
    _, qa_pipe, config = load_pipeline_components() 
    
  
    dev_df = pd.read_csv('data/cc.tsv', sep='\t')
    
    f1_metric = load_metric("f1")
    exact_match_metric = load_metric("exact_match")

    predictions = []
    references = []
    
    for _, example in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Evaluating on dataset"):
        multi_hop_question = example['Question']
        ground_truth_answers = ast.literal_eval(example['A1'])
        sub_questions = [example['Q1'], example['Q2']]
        
        sub_answers = []
        for sub_q in sub_questions:
            search_results = search_web(sub_q, config['pipeline']['max_search_results'])
            if not search_results.strip():
                continue
            
            try:
                qa_result = qa_pipe(question=sub_q, context=search_results)
                sub_answers.append(qa_result['answer'])
            except Exception as e:
                logging.error(f"Could not answer sub-question '{sub_q}': {e}")
        
        predicted_answer = combine_results(multi_hop_question, sub_answers)
        
        predictions.append({'id': str(example.name), 'prediction_text': predicted_answer})
        references.append({'id': str(example.name), 'answers': {'text': ground_truth_answers}})
        
    # Calculate metrics
    f1_result = f1_metric.compute(predictions=predictions, references=references)
    exact_match_result = exact_match_metric.compute(predictions=predictions, references=references)
    
    logging.info("\n--- Evaluation Results ---")
    logging.info(f"Exact Match: {exact_match_result['exact_match']:.4f}")
    logging.info(f"F1 Score: {f1_result['f1']:.4f}")

if __name__ == "__main__":
    evaluate_system()
