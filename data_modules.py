import os
import torch
from datasets import load_dataset
import random


def create_sciq_item(row, hint=False):
    question = f'Choose the correct alphabet of the answer from the following. Your answer should follow format of <alphabet>: <choice>. For example, "A: Answer".\nQuestion: {row["question"]}'
    support = f'Hint: {row["support"]}'
    answer_indexs = ["A", "B", "C", "D"]
    choices = [
        row["distractor1"],
        row["distractor2"],
        row["distractor3"],
        row["correct_answer"],
    ]
    choices = sorted(choices, key=lambda _: random.random())
    answer_idx = choices.index(row["correct_answer"])
    answer = "\n".join(
        [f"{answer_indexs[i]}: {choice}" for i, choice in enumerate(choices)]
    )

    inputs = (
        "\n".join([question, answer])
        if not hint
        else "\n".join([question, support, answer])
    )
    targets = f"{answer_indexs[answer_idx]}: {choices[answer_idx]}"

    return {
        "inputs": inputs,
        "targets": targets,
    }


def load_sciq_dataset(tokenizer, sequence_length=512):
    dataset = load_dataset("allenai/sciq")

    # count cpu cores
    num_proc = 8 if os.cpu_count() > 8 else os.cpu_count() - 1

    def create_item(row, hint=False):
        question = f'Question: {row["question"]}'
        support = f'Hint: {row["support"]}'
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
            "Choose the correct alphabet of the answer from the following.\n"
            + "\n".join(
                [f"{answer_indexs[i]}: {choice}" for i, choice in enumerate(choices)]
            )
        )

        inputs = (
            "\n".join([question, answer])
            if not hint
            else "\n".join([question, support, answer])
        )
        targets = answer_indexs[answer_idx]

        return {
            "inputs": inputs,
            "targets": targets,
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
            model_inputs["attention_mask"][i] = [0] * (
                max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (
                max_length - len(sample_input_ids)
            ) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:max_length]
            )
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(
        create_item,
        batched=False,
        num_proc=num_proc,
    )

    dataset = dataset.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": sequence_length,
        },
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    return dataset
