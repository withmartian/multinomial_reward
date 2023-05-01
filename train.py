import json

from transformers import TrainingArguments, AutoTokenizer, Trainer, T5EncoderModel
from src import Model, RankingDataset, create_comparison_dataset, DataCollator

MAX_RANKS_PER_BATCH = 2
MAX_LENGTH = 512


def get_training_examples():
    with open("./examples/reddit-tldr-train.json", "r") as f:
        return json.load(f)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
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
)

base_model = T5EncoderModel.from_pretrained("google/flan-t5-xl")

model = Model(
    base_model,
    tokenizer,
    max_ranks_per_batch=MAX_RANKS_PER_BATCH
)

layers = model.transformer.encoder.block
num_layers = len(layers)
num_unfrozen = int(0.3 * num_layers)
for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)

training_examples = get_training_examples()

train_pairs = create_comparison_dataset(training_examples)
train_dataset_false = RankingDataset(train_pairs[:2000], tokenizer, max_length=MAX_LENGTH, filter_uniform_rankings=False)
print("len() with FALSE: ", len(train_dataset_false))
train_dataset = RankingDataset(train_pairs[:2000], tokenizer, max_length=MAX_LENGTH)
print("len() with TRUE: ", len(train_dataset))
data_collator = DataCollator(max_ranks_per_batch=MAX_RANKS_PER_BATCH, max_sequence_length=MAX_LENGTH)

Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
).train()
