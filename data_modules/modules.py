import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class Formatter:
    def __init__(self, format="phi3", system_prompt=None):
        self.format = format
        self.format_fn = getattr(self, f"_format_{format}")
        self.system_prompt = (
            system_prompt
            if system_prompt
            else "You are the smartest student in class. Try your best answer correctly to questions."
        )

    def set_system_prompt(self, prompt) -> None:
        self.system_prompt = prompt

    def __call__(self, **kwargs) -> dict[str, str]:
        return self.format_fn(**kwargs)

    def _format_phi3(self, question, answer):
        input = ""
        if self.system_prompt:
            input += f"<|system|>\n{self.system_prompt}<|end|>\n"
        input += f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        label = f"{answer})<|end|>"
        input += label
        return {"input": input, "label": label, "answer": answer}

    def _format_llama3(self, question, answer):
        input = ""
        if self.system_prompt:
            input += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>]\n\n{self.system_prompt}<|eot_id|>"
        input += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        label = f"\n\n{answer})<|eot_id|>"
        input += label
        return {"input": input, "label": label, "answer": answer}


class SciQDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        max_seq_len=512,
        format="phi3",
        use_hint=False,
    ):
        assert split in [
            "train",
            "validation",
            "test",
        ], f"Invalid split provided : {split}."
        self.split = split
        self.tokenizer = tokenizer
        self.padding_side = "right" if "llama3" in format else "left"
        self.formatter = Formatter(format=format)
        self.max_seq_len = max_seq_len

        self.dataset = load_dataset("allenai/sciq", split=split)
        self.dataset = self.dataset.map(
            self._create_sciq_item,
            fn_kwargs={"hint": use_hint},
            batched=False,
            remove_columns=self.dataset.column_names,
            load_from_cache_file=False,
        ).with_format("torch")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _create_sciq_item(self, row, hint=False) -> dict[str, torch.Tensor | int]:
        # question = f'Choose the correct alphabet of the answer from the following. Your answer should follow format of <alphabet>: <choice>.\nQuestion: {row["question"]}'
        system = "You are the smartest student in class. Try your best answer correctly to questions."
        question = f'The following are multiple choice questions about science. \n **Question** {row["question"]}'
        if hint:
            question += f'\n **Hint** {row["support"]}'
        answer_indexs = ["A", "B", "C", "D"]
        choices = [
            row["distractor1"],
            row["distractor2"],
            row["distractor3"],
            row["correct_answer"],
        ]
        choices = sorted(choices, key=lambda _: random.random())
        answer_idx = choices.index(row["correct_answer"])
        choices = "\n".join(
            [f"({answer_indexs[i]}) {choice}" for i, choice in enumerate(choices)]
        )
        question += f"\n **Choices** \n{choices} \n **Answer:** ("
        answer = answer_indexs[answer_idx]

        item = self.formatter(
            question=question,
            answer=answer,
        )

        tk_item = self._tokenize_item(item)

        return tk_item

    def _tokenize_item(self, item) -> dict[str, torch.Tensor | int]:
        tk_question = self.tokenizer(
            item["input"],
            return_tensors="pt",
        )
        tk_answer = self.tokenizer(
            item["label"],
            return_tensors="pt",
        )

        tk_answer = tk_answer["input_ids"][:, 1:]  # truncate statr_of_text token
        label = torch.ones_like(tk_question["input_ids"]) * -100
        label[:, -tk_answer.size(1) :] = tk_answer

        # append padding to max_seq_len
        for key in tk_question:
            tk_question[key] = torch.nn.functional.pad(
                tk_question[key],
                (
                    (0, self.max_seq_len - tk_question[key].size(1))
                    if self.padding_side == "right"
                    else (self.max_seq_len - tk_question[key].size(1), 0)
                ),
                value=self.tokenizer.pad_token_id,
            )
        label = torch.nn.functional.pad(
            label,
            (
                (0, self.max_seq_len - label.size(1))
                if self.padding_side == "right"
                else (self.max_seq_len - label.size(1), 0)
            ),
            value=-100,
        )

        return {
            "input_ids": tk_question["input_ids"].squeeze(),
            "attention_mask": tk_question["attention_mask"].squeeze(),
            "labels": label.squeeze(),
        }


class ScienceQADataset:
    def __init__(self, df, shot_qids, args, shot_df, tokenizer):
        super().__init__()
        self.prompts = []

        shot_added_df = pd.concat((df, shot_df))
        shot_added_dict = shot_added_df.transpose().to_dict()

        for qid in tqdm([str(x) for x in df.index if x not in shot_qids]):
            prompt = build_prompt(shot_added_dict, shot_qids, qid, args)

            self.prompts.append(prompt)
        self.answers = df.progress_apply(
            lambda x: x["choices"][x["answer"]], axis=1
        ).to_list()
        self.distractors = df.progress_apply(
            lambda x: ";".join(
                [
                    choice
                    for choice in x["choices"]
                    if choice is not x["choices"][x["answer"]]
                ]
            ),
            axis=1,
        ).to_list()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index], self.answers[index], self.distractors[index]
