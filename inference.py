import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch
import tqdm
from accelerate import infer_auto_device_map
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data_modules import load_dataset
from utils import calc_accuracy

MODEL_DICT = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def generate(model, loader, tokenizer, device="cuda"):
    model_outputs = {"targets": [], "preds": []}
    for batch in tqdm.tqdm(loader):
        batch = {
            k: v.to(device) for k, v in batch.items()
        }  # keys : inputs, attention_mask, labels

        output = model.generate(**batch)
        model_outputs["targets"].extend(
            tokenizer.batch_decode(output, skip_special_tokens=True)
        )
        model_outputs["preds"].extend(
            tokenizer.batch_decode(output, skip_special_tokens=True)
        )
    return model_outputs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Load dataset
    val_dset = load_dataset(
        args.dataset,
        split="validation",
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        format=args.model_name,
        use_hint=args.use_hint,
    )
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DICT[args.model_name], **model_kwargs
    )

    with torch.no_grad():
        loss, outputs = generate(model, val_loader, tokenizer, device="cuda")
        accuracy = calc_accuracy(outputs)
    print(f"[Validation]\n- Loss: {loss}\n- Accuracy: {accuracy}")

    # Save outputs
    output_dir = f"{args.output_dir}/{args.model_name}_{args.dataset}"
    if args.use_hint:
        output_dir += "_hint"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/baseline.pkl", "wb") as f:
        pkl.dump(
            {
                "predictions": outputs,
                "loss": loss,
                "accuracy": accuracy,
            },
            f,
        )


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs")

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
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--soft_prompt_length", type=int, default=64)

    # Training/Validation arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)

    # Inferencing arguments
    parser.add_argument("--use_hint", action="store_true")

    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)