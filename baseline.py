import pickle as pkl
import random

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

from data_modules import create_sciq_item

SEED = 42
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def main():
    # Load dataset
    dataset = load_dataset("allenai/sciq")
    format_dataset = dataset.map(
        create_sciq_item,
        fn_kwargs={"hint": False},
        batched=False,
        num_proc=8,
    )

    # Load pre-trained model
    model_kwargs = {
        "use_cache": False,
        "trust_remote_code": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 10,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    format_message = lambda x: [
        {
            "role": "system",
            "content": "Answer correctly to following question. Try your best to reduce hallucination.",
        },
        {"role": "user", "content": x},
    ]

    logs = {"Accuracy": 0, "FP": []}
    num_correct = 0
    num_total = 0
    for i in tqdm.trange(len(format_dataset["validation"])):
        input = format_message(format_dataset["validation"][i]["inputs"])
        target = format_dataset["validation"][i]["targets"].lower()
        output = pipe(input, **generation_args)[0]["generated_text"]
        parsed_output = output[1].lower()
        if parsed_output not in ["a", "b", "c", "d"]:
            print(f"Index {i} : {output}\nTarget : {target}")
        if target[0] == parsed_output:
            num_correct += 1
        else:
            logs["FP"].append((input, output))
        num_total += 1

    print(f"Accuracy : {num_correct/num_total*100:2.2f}")
    logs["accuracy"] = num_correct / num_total * 100
    with open("results/phi3_sciq.pkl", "wb") as f:
        pkl.dump(logs, f)
    return 0


if __name__ == "__main__":
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    main()
