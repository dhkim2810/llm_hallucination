import os
import random

import torch
from peft import (
    PeftType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_config,
    get_peft_model,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from datasets import load_dataset

BASE_DIR = os.environ.get("SLRUM_TMPDIR", "tmp")

LR = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
SOFT_PROMPT_LENGTH = 64
DEVICE = "cuda"

MODEL = "microsoft/Phi-3-mini-4k-instruct"


def create_item(row):
    question = row["question"]
    support = row["support"]
    answer_indexs = ["A", "B", "C", "D"]
    choices = [
        row["distractor1"],
        row["distractor2"],
        row["distractor3"],
        row["correct_answer"],
    ]
    choices = sorted(choices, key=lambda k: random.random())
    answer_idx = choices.index(row["correct_answer"])
    answer = answer_indexs[answer_idx]  # + ". " + row['correct_answer']

    return {
        "sentence": f"Question: {question}",
        "answers": "Choose the correct alphabet of the answer from the following.\n"
        + "\n".join(
            [f"{answer_indexs[i]}: {choice}" for i, choice in enumerate(choices)]
        ),
        "label": answer_idx,
        "text_label": answer,
        "support": f"Context: {support}",
    }


def preprocess_function(data, tokenizer, support, max_length):
    inputs = (
        data["sentence"] + "\n" + data["answers"]
        if not support
        else data["sentence"] + "\n" + data["support"] + "\n" + data["answers"]
    )
    targets = data["text_label"]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


dataset = load_dataset("allenai/sciq")
dataset = dataset.map(
    create_item,
    batched=False,
    num_proc=8,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"

processed_datasets = dataset.map(
    preprocess_function,
    fn_kwargs={
        "tokenizer": tokenizer,
        "support": False,
        "max_length": 512,
    },
    batched=False,
    num_proc=8,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=BATCH_SIZE,
    pin_memory=True,
)
eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=default_data_collator,
    batch_size=BATCH_SIZE,
    pin_memory=True,
)


peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=SOFT_PROMPT_LENGTH,
    prompt_tuning_init_text="Answer correctly to following question. Try your best to reduce hallucination.",
    tokenizer_name_or_path=MODEL,
)

model = AutoModelForCausalLM.from_pretrained(MODEL)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
)


model = model.to(DEVICE)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(
                torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                skip_special_tokens=True,
            )
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

# Save model
model.save_pretrained(os.path.join(BASE_DIR, "ckpt"))
