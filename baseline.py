import numpy as np
import torch
import tqdm
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from data_modules import create_sciq_item

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
        "device_map": "cuda",
        "trust_remote_code": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
    }

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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

    num_correct = 0
    num_total = 0
    for i in tqdm.trange(len(format_dataset["validate"])):
        output = pipe(
            format_message(format_dataset["validate"][i]["inputs"]), **generation_args
        )[0]["generated_text"]
        parsed_output = output[1].lower()
        if parsed_output not in ["a", "b", "c", "d"]:
            print(
                f"Index {i} : {output}\nTarget : {format_dataset['train'][i]['targets']}"
            )
        if format_dataset["train"][i]["targets"].lower() == parsed_output:
            num_correct += 1
        num_total += 1

    print(f"Accuracy : {num_correct/num_total*100:2.2f}")
    return 0


if __name__ == "__main__":
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    main()
