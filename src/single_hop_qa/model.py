
def load_qa_model(pretrained_model_name):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name)
    return tokenizer, model

