import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch
import tqdm
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
)

from data_modules import load_dataset
from utils import calc_accuracy
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer, scoring

MODEL_DICT = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

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


def calculate_bleu(preds, refs):
    references = [[ref.split()] for ref in refs]
    candidates = [pred.split() for pred in preds]
    score = corpus_bleu(references, candidates)
    return score


def calculate_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for ref, pred in zip(refs, preds):
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return result


def generate(model, loader, tokenizer, device="cuda"):
    model_outputs = {"answers": [], "labels": [], "preds": []}
    for batch in tqdm.tqdm(loader):
        batch_size = len(batch["answer"])
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        output = model.generate(**inputs, max_new_tokens=2).detach().cpu()
        residual = output.size(-1) - batch["labels"].size(-1)
        mask = torch.cat(
            [(batch["labels"] != -100), torch.ones(batch_size, residual)], dim=-1
        ).long()
        model_outputs["answers"].extend(batch["answer"])
        model_outputs["labels"].extend(
            tokenizer.batch_decode(
                torch.clip(batch["labels"], 0), skip_special_tokens=True
            )
        )
        model_outputs["preds"].extend(
            tokenizer.batch_decode(output * mask, skip_special_tokens=True)
        )
    return model_outputs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DICT[args.model_name] if args.model_path is None else args.model_path
    )
    if args.model_name == "phi3":
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif args.model_name == "llama3":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError("Invalid model name")

    # Load dataset
    test_dset = load_dataset(
        args.dataset,
        tokenizer=tokenizer,
        split="test",
        model=args.model_name,
        prompt_format=args.prompt_format,
        max_length=args.max_length,
        use_hint=args.tune == "hint",
        padding_side="left" if args.model_name == "phi3" else "right",
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    # Load pre-trained model
    model_kwargs = {
        "token": os.environ["HF_TOKEN"],
        "use_cache": True,
        "trust_remote_code": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
        "attn_implementation": "flash_attention_2",
        "device_map": args.device,
    }

    if args.model_path is None or args.tune == "prompt":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DICT[args.model_name],
            **model_kwargs,
        )
        if args.tune == "prompt":
            peft_config = PeftConfig.from_pretrained(args.model_path)
            model = PeftModel.from_pretrained(model, args.model_path)
            model.load_adapter(args.model_path, adapter_name="default")
            model.set_adapter("default")
    elif args.model_path is not None and args.tune != "prompt":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs,
        )

    with torch.no_grad():
        outputs = generate(model, test_loader, tokenizer, device="cuda")
        # accuracy = calc_accuracy(outputs)

    # Calculate BLEU and ROUGE scores
    bleu_score = calculate_bleu(outputs["preds"], outputs["labels"])
    rouge_scores = calculate_rouge(outputs["preds"], outputs["labels"])

    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"ROUGE Scores: {rouge_scores}")

    # Save outputs
    with open(
        f"gen_results/{args.model_name}_{args.dataset}_{args.tune}.pkl", "wb"
    ) as f:
        pkl.dump(outputs, f)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--tune",
        type=str,
        default="baseline",
        choices=["baseline", "weight", "prompt", "hint"],
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="phi3")
    parser.add_argument("--model_path", type=str, default=None)

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sciq", "scienceqa"],
    )
    parser.add_argument("--max_length", type=int, default=128)

    # Inferencing arguments
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
