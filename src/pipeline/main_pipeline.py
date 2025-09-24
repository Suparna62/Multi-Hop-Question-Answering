import yaml
from transformers import pipeline
from duckduckgo_search import DDGS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pipeline_components():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    decomposition_pipe = pipeline(
        "text2text-generation",
        model=config['decomposition_model']['output_dir'],
        tokenizer=config['decomposition_model']['output_dir']
    )

    qa_pipe = pipeline(
        "question-answering",
        model=config['single_hop_qa_model']['output_dir'],
        tokenizer=config['single_hop_qa_model']['output_dir']
    )
    return decomposition_pipe, qa_pipe, config

def search_web(query, max_results):
    logging.info(f"Searching web for: {query}")
    with DDGS() as ddgs:
        results = [r['body'] for r in ddgs.text(query, max_results=max_results)]
    return " ".join(results)

def combine_results(question, sub_answers):
    
    if not sub_answers:
        return "Not enough information to combine answers."
    
 
    combined_context = f"Sub-answers: {'; '.join(sub_answers)}. "

    try:
        final_qa_result = qa_pipe(question=question, context=combined_context)
        return final_qa_result['answer']
    except Exception as e:
        logging.error(f"Error during final combination: {e}")
        return "Error in combining answers."


def run_multi_hop_pipeline(multi_hop_question, decomposition_pipe, qa_pipe, config):
    logging.info(f"Original Question: {multi_hop_question}")

    # Stage 1: Decompose question
    logging.info("Stage 1: Decomposing question...")
    decomposed_output = decomposition_pipe(f"decompose question: {multi_hop_question}")[0]['generated_text']
    sub_questions = [q.strip() for q in decomposed_output.split(';') if q.strip()]
    logging.info(f"Generated Sub-questions: {sub_questions}")

    if not sub_questions:
        logging.warning("Failed to decompose. Using original question for a single search.")
        sub_questions = [multi_hop_question]

    # Stage 2: Answer sub-questions individually
    sub_answers = []
    for sub_q in sub_questions:
        search_results = search_web(sub_q, config['pipeline']['max_search_results'])
        if not search_results.strip():
            logging.warning(f"No search results for sub-question: {sub_q}")
            continue

        try:
            qa_result = qa_pipe(question=sub_q, context=search_results)
            sub_answers.append(qa_result['answer'])
            logging.info(f"Answer for '{sub_q}': {qa_result['answer']}")
        except Exception as e:
            logging.error(f"Could not answer sub-question '{sub_q}': {e}")
    
    # Stage 3: Combine results
    logging.info("Stage 3: Combining results...")
    final_answer = combine_results(multi_hop_question, sub_answers)
    logging.info(f"Final Combined Answer: {final_answer}")
    return final_answer

if __name__ == "__main__":
    decomposition_pipe, qa_pipe, config = load_pipeline_components()
    question = "What is the capital of the birthplace of Rumi?"
    run_multi_hop_pipeline(question, decomposition_pipe, qa_pipe, config)

