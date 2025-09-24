import yaml
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from src.decomposition.model import load_decomposition_model
import evaluate

def train_decomposition_model():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['decomposition_model']
    tokenizer, model = load_decomposition_model(model_config['pretrained_model'])
    
    dataset = load_dataset('json', data_files={
        'train': 'data/processed/train_decomposed.jsonl',
        'validation': 'data/processed/dev_decomposed.jsonl'
    })
    
    def preprocess_function(examples):
        inputs = [f"decompose question: {q}" for q in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(text_target=examples["sub_questions"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[tokenizer.decode(label, skip_special_tokens=True)] for label in labels]
        
        result = metric.compute(predictions=decoded_preds, references=labels, use_stemmer=True)
        return {k: v * 100 for k, v in result.items()}
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_config['output_dir'],
        per_device_train_batch_size=model_config['batch_size'],
        per_device_eval_batch_size=model_config['eval_batch_size'],
        learning_rate=model_config['learning_rate'],
        num_train_epochs=model_config['num_train_epochs'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    train_decomposition_model()
