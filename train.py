import argparse
import os
import random

import numpy as np
import torch
from accelerate import PartialState
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    default_data_collator,
)
from trl import SFTTrainer

from data_modules import load_dataset

MODEL_DICT = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# PEFT configuration
## Phi3 Fine-tuning(linear probing)
PEFT_FT_DICT = {
    "phi3": LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        modules_to_save=None,
        task_type=TaskType.CAUSAL_LM,
    ),
    "llama3": LoraConfig(
        r=8,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    ),
}

## Phi3 Prompt-tuning(Soft Prompt)
PEFT_PT_DICT = {
    "phi3": PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=64,
        tokenizer_name_or_path=MODEL_DICT["phi3"],
        prompt_tuning_init_text="You are the smartest student in class. Try your best answer correctly to questions.",
    ),
    "llama3": PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=64,
        tokenizer_name_or_path=MODEL_DICT["llama3"],
        prompt_tuning_init_text="You are the smartest student in class. Try your best answer correctly to questions.",
    ),
}

PEFT_DICT = {
    "weight": PEFT_FT_DICT,
    "prompt": PEFT_PT_DICT,
}


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.unk_token

    # Load dataset
    train_dset = load_dataset(
        args.dataset,
        split="train",
        tokenizer=tokenizer,
        format=args.model_name,
        max_length=args.max_length,
        use_hint=args.use_hint,
        padding_side="left" if args.model_name == "phi3" else "right",
    )
    val_dset = load_dataset(
        args.dataset,
        split="validation",
        tokenizer=tokenizer,
        format=args.model_name,
        max_length=args.max_length,
        use_hint=args.use_hint,
        padding_side="left" if args.model_name == "phi3" else "right",
    )

    # Load pre-trained model
    model_kwargs = {
        "token": os.environ["HF_TOKEN"],
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
        "attn_implementation": "flash_attention_2",
        "device_map": {"": args.device},
        "trust_remote_code": True,
        "use_cache": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DICT[args.model_name], **model_kwargs
    )

    # sft_config = TrainingArguments(
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=4,
    #     warmup_steps=2,
    #     num_train_epochs=args.num_epochs,
    #     learning_rate=args.lr,
    #     fp16=True,
    #     logging_steps=1,
    #     output_dir=args.output_dir,
    #     optim="paged_adamw_8bit",
    #     use_tqdm=True,
    # )

    training_config = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 1,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": args.num_epochs,
        "max_steps": -1,
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": args.batch_size,
        "per_device_train_batch_size": args.batch_size,
        "remove_unused_columns": True,
        "save_steps": 1000,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
    }
    sft_config = TrainingArguments(**training_config)

    peft_params = PEFT_DICT[args.tune][args.model_name]

    trainer = SFTTrainer(
        args=sft_config,
        peft_config=peft_params,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dset,
        eval_dataset=val_dset,
        max_seq_length=args.max_length,
        dataset_text_field="input_ids",
        # dataset_label_field="labels",
        packing=True,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(val_dset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save the trained model
    trainer.save_model(sft_config.output_dir)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Fine-tuning methods
    parser.add_argument(
        "--tune", type=str, default="weight", choices=["weight", "prompt"]
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["phi3", "llama3"],
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sciq", "scienceqa"],
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--soft_prompt_length", type=int, default=64)

    # Training/Validation arguments
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)

    # Inferencing arguments
    parser.add_argument("--use_hint", action="store_true")

    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Set device
    args.device = PartialState().process_index

    # Set output directory
    args.output_dir = f"{args.output_dir}/{args.model_name}_{args.dataset}"
    if args.use_hint:
        args.output_dir += "_hint"
    args.output_dir += f"/{args.tune}"

    main(args)
