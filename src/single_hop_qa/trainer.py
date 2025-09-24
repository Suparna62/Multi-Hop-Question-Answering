import yaml
from transformers import TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
from src.single_hop_qa.model import load_qa_model
from tqdm.auto import tqdm
import torch

def train_qa_model():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    qa_model_config = config['single_hop_qa_model']
    qa_tokenizer, qa_model = load_qa_model(qa_model_config['pretrained_model'])
    
    
    squad_dataset = load_dataset("squad")
    
    max_length = config['pipeline']['max_context_length']
    stride = 128
    
    def prepare_train_features(examples):
        tokenized_examples = qa_tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = tokenized_examples.pop("offset_mapping")
        sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
    
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            sample_index = sample_map[i]
            answers = examples["answers"][sample_index]
            start_char = answers["answer_start"][0] if answers["answer_start"] else -1
            text = answers["text"][0] if answers["text"] else ""
            end_char = start_char + len(text)
            
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
    
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
    
            if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
            
        return tokenized_examples
    
    tokenized_squad = squad_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=squad_dataset["train"].column_names,
    )
    
    training_args = TrainingArguments(
        output_dir=qa_model_config['output_dir'],
        evaluation_strategy="epoch",
        learning_rate=qa_model_config['learning_rate'],
        per_device_train_batch_size=qa_model_config['batch_size'],
        per_device_eval_batch_size=qa_model_config['eval_batch_size'],
        num_train_epochs=qa_model_config['num_train_epochs'],
        weight_decay=0.01,
        report_to="none",
    )
    
    trainer = Trainer(
        qa_model,
        training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        tokenizer=qa_tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    train_qa_model()

