import json

from transformers import TrainingArguments, AutoTokenizer, Trainer, T5EncoderModel, LlamaTokenizer, LlamaModel
from src import Model, RankingDataset, create_comparison_dataset, DataCollator
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

MAX_RANKS_PER_BATCH = 2
MAX_LENGTH = 512


def get_training_examples():
    data_path = "CarperAI/openai_summarize_comparisons"
    dataset = load_dataset(data_path)
    return remap_chosen_rejected_to_ranking(dataset["train"])


def get_validation_examples():
    data_path = "CarperAI/openai_summarize_comparisons"
    dataset = load_dataset(data_path)
    return remap_chosen_rejected_to_ranking(dataset["valid1"])


def remap_chosen_rejected_to_ranking(dataset):
    result = []
    for sample in tqdm(dataset):
        result.append({
            "prompt": sample["prompt"],
            "ranked_outputs": [
                sample["chosen"],
                sample["rejected"]
            ]
        })
    return result


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds_flat = [item for sublist in preds for item in sublist]
    end_scores = [pred.squeeze() for pred in preds_flat]

    result = {}
    total_pairs = 0
    correctly_ranked_pairs = 0

    for i in range(len(end_scores)):
        for j in range(i + 1, len(end_scores)):
            total_pairs += len(end_scores[i])
            correctly_ranked_pairs += sum(
                end_scores[i][k] > end_scores[j][k] for k in range(len(end_scores[i]))
            )

    if total_pairs > 0:
        acc = correctly_ranked_pairs / total_pairs
    else:
        acc = 0

    result["accuracy"] = acc
    return result


def get_T5_training_settings(_params, _dataset):
    training_args = TrainingArguments(
        output_dir="./output",
        # num_train_epochs=10,
        num_train_epochs=1,
        logging_steps=100,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        learning_rate=1e-5,
        save_total_limit=1,
        eval_steps=100,
        evaluation_strategy="steps"
    )
    return training_args, None, None


def get_llama_training_settings(params, dataset):
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        logging_steps=100,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=3000,
        warmup_steps=100,
        logging_dir="./logs",
        eval_steps=100,
        evaluation_strategy="steps"
    )
    optimizer = AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, len(dataset), eta_min=3e-5)
    return training_args, optimizer, scheduler


def freeze(layers, unfrozen_percent=0.3):
    num_layers = len(layers)
    num_unfrozen = int(unfrozen_percent * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)
    return layers


def get_T5_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    base_model = T5EncoderModel.from_pretrained("google/flan-t5-xl")
    freeze(base_model.encoder.block)
    return tokenizer, base_model


def get_llama_model():
    tokenizer = LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf",
        device_map="auto"
    )
    base_model = LlamaModel.from_pretrained(
        "decapoda-research/llama-7b-hf",
        torch_dtype=torch.float32
    )

    new_pad_token = "<pad>"
    tokenizer.add_tokens([new_pad_token])
    tokenizer.pad_token = new_pad_token
    base_model.resize_token_embeddings(len(tokenizer))

    freeze(base_model.layers)

    return tokenizer, base_model


tokenizer, base_model = get_llama_model()

model = Model(
    base_model,
    tokenizer,
    max_ranks_per_batch=MAX_RANKS_PER_BATCH
)

training_examples = get_training_examples()

train_pairs = create_comparison_dataset(get_training_examples())
train_dataset = RankingDataset(train_pairs[400:], tokenizer, max_length=MAX_LENGTH)
eval_pairs = create_comparison_dataset(get_validation_examples())
eval_dataset = RankingDataset(eval_pairs[:100], tokenizer, max_length=MAX_LENGTH)
data_collator = DataCollator(max_ranks_per_batch=MAX_RANKS_PER_BATCH, max_sequence_length=MAX_LENGTH)

training_args, optimizer, scheduler = get_llama_training_settings(model.parameters(), train_dataset)

Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
).train()
