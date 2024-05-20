import os
import random
import pickle

import torch
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from datasets import load_dataset

BASE_DIR = os.environ.get("SLRUM_TMPDIR", "tmp")

LR = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
SEQ_LENGTH = 512
SOFT_PROMPT_LENGTH = 64
DEVICE = "cuda"

MODEL = "microsoft/Phi-3-mini-4k-instruct"


def create_item(row, support=False):
    question = f'Question: {row["question"]}'
    support = f'Context : {row["support"]}'
    answer_indexs = ["A", "B", "C", "D"]
    choices = [
        row["distractor1"],
        row["distractor2"],
        row["distractor3"],
        row["correct_answer"],
    ]
    choices = sorted(choices, key=lambda k: random.random())
    answer_idx = choices.index(row["correct_answer"])
    answer = "Choose the correct alphabet of the answer from the following.\n" + "\n".join([f"{answer_indexs[i]}: {choice}" for i, choice in enumerate(choices)])

    inputs = (
        "\n".join([question, answer])
        if not support
        else "\n".join([question, support, answer])
    )
    targets = answer_indexs[answer_idx]

    return {
        "inputs" : inputs,
        "targets" : targets,
    }

def preprocess_function(data, tokenizer, max_length):
    batch_size = len(data["inputs"])
    inputs = [f"Inputs : {x} Answer: " for x in data["inputs"]]
    targets = [x for x in data["targets"]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset = load_dataset("allenai/sciq")
dataset = dataset.map(
    create_item,
    batched=False,
    num_proc=8,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"

processed_datasets = dataset.map(
    preprocess_function,
    fn_kwargs={
        "tokenizer": tokenizer,
        "max_length": SEQ_LENGTH,
    },
    batched=True,
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

# Quantization and Model Configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)

# Load pre-trained model
model_kwargs = {
    "use_cache": False,
    "trust_remote_code": True,
    "attn_implementation": "flash_attention_2",
    "quantization_config": quant_config,
    "device_map": {"": 0},
}

model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())


# configuration = Phi3Config.from_pretrained(MODEL, trust_remote_code=True)
# model = Phi3Model(configuration)
# model = get_peft_model(model, peft_config)
# print(model.print_trainable_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
)


model = model.to(DEVICE)

losses = {'train': [], 'val': []}

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

    losses['train'].append((train_epoch_loss, train_ppl))
    losses['val'].append((eval_epoch_loss, eval_ppl))

# Save model
model.save_pretrained(os.path.join(BASE_DIR, "ckpt"))
with open(os.path.join(BASE_DIR, "result.pkl"), "wb") as f:
    pickle.dump(losses, f)
