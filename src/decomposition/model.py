from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_decomposition_model(model_name_or_path="t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return tokenizer, model
