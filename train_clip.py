from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers import Trainer, TrainingArguments, DataCollator
from src_website_quality import Model, DataCollator, ImageRankingDataset

import torch

def get_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    base_model.to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", padding=True)
    return base_model, processor

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
                end_scores[i][k] > end_scores[j][k] for k in range(len(end_scores[i])))

    if total_pairs > 0:
        acc = correctly_ranked_pairs / total_pairs
    else:
        acc = 0

    result["accuracy"] = acc
    return result


if __name__ == "__main__":
    base_model, processor = get_clip_model()
    model = Model(base_model, processor, max_ranks_per_batch=27)
    dataset = [] # FIXME: get dataset, probably using GCS
    train_dataset = ImageRankingDataset(dataset[:40], processor)
    eval_dataset = ImageRankingDataset(dataset[40:], processor)
    data_collator = DataCollator()

    # FIXME: change training args
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        logging_steps=1,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        per_device_train_batch_size=1, # Need to be 1 since tensors are not the same size
        per_device_eval_batch_size=1, # Need to be 1
        save_steps=50,
        warmup_steps=10,
        logging_dir="./logs",
        learning_rate=1e-5,
        save_total_limit=1,
        do_eval=True,
        eval_steps=1,
        evaluation_strategy="steps",
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
    )

    trainer.train()