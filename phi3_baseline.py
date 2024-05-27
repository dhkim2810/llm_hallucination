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
    question = f'"Question: {row["question"]}'
    support = f'"Context : {row["support"]}"'
    answer_indexs = ["A", "B", "C", "D"]
    choices = [
        row["distractor1"],
        row["distractor2"],
        row["distractor3"],
        row["correct_answer"],
    ]
    choices = sorted(choices, key=lambda _: random.random())
    answer_idx = choices.index(row["correct_answer"])
    answer = (
        'Choose the correct alphabet of the answer from the following".\n\n'
        + "\n".join(
            [f"{answer_indexs[i]}: {choice}" for i, choice in enumerate(choices)]
        )
    )
    inputs = (
        "\n".join([question, answer])
        if not support
        else "\n".join([support, question, answer])
    ) + "\n Answer: "
    targets = answer_indexs[answer_idx]
    return {
        "inputs": inputs,
        "targets": targets,
    }


def preprocess_function(data, tokenizer, pre_tokens, max_length):
    batch_size = len(data["inputs"])

    user_prompt = [x for x in data["inputs"]]
    targets = [x for x in data["targets"]]

    model_inputs = tokenizer(user_prompt)
    labels = tokenizer(targets)

    for i in range(batch_size):
        user_input = torch.Tensor(tokenizer(data["inputs"][i])[1:]).requires_grad_(
            False
        )
        sample_input_ids = torch.cat(
            [
                pre_tokens["sys"],
                pre_tokens["system_token"],
                pre_tokens["end"],
                pre_tokens["user"],
                user_input,
                pre_tokens["end"],
                pre_tokens["assist"],
            ]
        )
        label_input_ids = torch.Tensor(
            labels["input_ids"][i] + [tokenizer.pad_token_id]
        )
        model_inputs["input_ids"][i] = torch.cat([sample_input_ids, label_input_ids])
        model_inputs["attention_mask"][i] = torch.ones_like(
            model_inputs["input_ids"][i]
        )
        labels["input_ids"][i] = torch.cat(
            [-100 * torch.ones_like(sample_input_ids), label_input_ids]
        )
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]

        pad_length = max(0, max_length - len(sample_input_ids))
        if pad_length:
            model_inputs["input_ids"][i] = torch.cat(
                [
                    tokenizer.pad_token_id * torch.ones(pad_length),
                    sample_input_ids,
                ]
            )
            model_inputs["attention_mask"][i] = torch.cat(
                [
                    torch.zeros(pad_length),
                    model_inputs["attention_mask"][i],
                ]
            )
            labels["input_ids"][i] = torch.cat(
                [
                    -100 * torch.ones(pad_length),
                    label_input_ids,
                ]
            )
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][
            :max_length
        ]
        labels["input_ids"][i] = labels["input_ids"][i][:max_length]
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

sys_indicator = torch.Tensor(tokenizer("<|system|>")["input_ids"][1:]).requires_grad_(
    False
)
user_indicator = torch.Tensor(tokenizer("<|user|>")["input_ids"][1:]).requires_grad_(
    False
)
assist_indicator = torch.Tensor(
    tokenizer("<|assistant|>")["input_ids"][1:]
).requires_grad_(False)
end_indicator = torch.Tensor(tokenizer("<|end|>")["input_ids"][1:]).requires_grad_(
    False
)
sys_token = torch.Tensor(
    tokenizer(
        "Answer correctly to following question. Try your best to reduce hallucination."
    )["input_ids"][1:]
).requires_grad_(False)

processed_datasets = dataset.map(
    preprocess_function,
    fn_kwargs={
        "tokenizer": tokenizer,
        "max_length": SEQ_LENGTH,
        "pre_tokens": {
            "sys": sys_indicator,
            "user": user_indicator,
            "assist": assist_indicator,
            "end": end_indicator,
            "system_token": sys_token,
        },
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
    bnb_4bit_use_double_quant=False,
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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
)

model = model.to(DEVICE)

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
    print(f"{eval_ppl=} {eval_epoch_loss=}")
