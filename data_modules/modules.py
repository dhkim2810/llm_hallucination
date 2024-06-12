import random

import torch
from datasets import Dataset, load_dataset

# from torch.utils.data import Dataset


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

    def _format_phi3(self, question, answer, is_generate=False):
        input = ""
        if self.system_prompt:
            input += f"<|user|> {self.system_prompt} {question}<|end|><|assistant|>"
        else:
            input += f"<|user|> {question}<|end|><|assistant|>"
        label = f"{answer})<|end|>"
        if not is_generate:
            input += label
        return {"input": input, "label": label, "answer": answer}

    def _format_llama3(self, question, answer, is_generate=False):
        input = ""
        if self.system_prompt:
            input += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>]\n\n{self.system_prompt}<|eot_id|>"
        input += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        label = f"{answer})<|eot_id|>"
        if not is_generate:
            input += label
        return {"input": input, "label": label, "answer": answer}


class FineTuneDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        split="train",
        format="phi3",
        max_length=256,
        use_hint=False,
        is_label=True,  # question with answers
        is_generation=False,  # question tokens without answer
        padding_side="right",
    ):
        self._split = split
        self.tokenizer = tokenizer
        self.formatter = Formatter(format=format)
        self.max_length = max_length
        self.is_generation = is_generation
        self.use_hint = use_hint
        self.padding_side = padding_side  # "right" if "llama3" in format else "left"
        self.is_label = is_label

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
        pass

    def _pad(self, tk_item: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(
            tk_item,
            (
                (0, self.max_length - tk_item.size(1))
                if self.padding_side == "right"
                else (self.max_length - tk_item.size(1), 0)
            ),
            value=self.tokenizer.pad_token_id,
        )

    def _tokenize_item(self, question, answer) -> dict[str, torch.Tensor | int]:
        tk_question = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        for key in tk_question:
            tk_question[key] = self._pad(tk_question[key])

        if not self.is_label:
            return {
                "input_ids": tk_question["input_ids"].squeeze(),
                "attention_mask": tk_question["attention_mask"].squeeze(),
            }

        tk_answer = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        tk_answer = tk_answer["input_ids"][:, 1:]  # truncate statr_of_text token
        label = torch.ones_like(tk_question["input_ids"]) * -100
        label[:, -tk_answer.size(1) :] = tk_answer
        label = self._pad(label)

        return {
            "input_ids": tk_question["input_ids"].squeeze(),
            "attention_mask": tk_question["attention_mask"].squeeze(),
            "labels": label.squeeze(),
        }


class SciQDataset(FineTuneDataset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = load_dataset("allenai/sciq", split=self._split)
        self.dataset = self.dataset.map(
            self._create_sciq_item,
            fn_kwargs={"hint": self.use_hint},
            batched=False,
            remove_columns=self.dataset.column_names,
            load_from_cache_file=False,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def reset(self):
        self.dataset = self.dataset.map(
            self._create_sciq_item,
            fn_kwargs={"hint": self.use_hint},
            batched=False,
            remove_columns=self.dataset.column_names,
            load_from_cache_file=False,
        )

    def _create_sciq_item(self, example, hint=False) -> dict[str, torch.Tensor | int]:
        # parse item
        question, question_with_answer, answer = self._parse_row(example, hint)

        # create message
        if self.is_generation:
            item = self.formatter(question=question, answer=answer, is_generate=True)
            tk_item = self.tokenizer(
                item["input"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            tk_item = self._pad(tk_item)
            return {"inputs": tk_item, "answer": answer}
        else:
            item = self.formatter(question=question, answer=answer)
            return self._tokenize_item(question, answer)

    def _parse_row(self, row, hint=False) -> tuple[str, str]:
        question = f'The following are multiple choice questions about science.\n**Question** {row["question"]}'
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
        question += f"\n**Choices** \n{choices}\n**Answer:** ("
        question_with_answer = question + answer_indexs[answer_idx] + ")"
        answer = answer_indexs[answer_idx]

        return question, question_with_answer, answer


class ScienceQADataset:
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = load_dataset("allenai/sciq", split=self._split)
        self.dataset = self.dataset.filter(lambda x: x["image"] == None)
        self.dataset = self.dataset.map(
            self._create_scienceqa_item,
            fn_kwargs={"hint": self.use_hint},
            batched=False,
            remove_columns=self.dataset.column_names,
            load_from_cache_file=False,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _create_scienceqa_item(
        self, example, hint=False
    ) -> dict[str, torch.Tensor | int]:
        # parse item
        question, question_with_answer, answer = self._parse_row(example, hint)

        # create message
        if self.is_generation:
            item = self.formatter(question=question, answer=answer, is_generate=True)
            tk_item = self.tokenizer(
                item["input"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            tk_item = self._pad(tk_item)
            return {"inputs": tk_item, "answer": answer}
        else:
            item = self.formatter(question=question, answer=answer)
            return self._tokenize_item(question, answer)

    def _parse_row(self, row, hint=False) -> tuple[str, str]:
        context = self.get_context_text(row, use_caption=False)
        question = self.get_question_text(row)
        choice = self.get_choice_text(row)
        answer = self.get_answer(row)

        input = (
            f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
            if hint
            else f"Question: {question}\nOptions: {choice}\n"
        )
        answer = f"Answer: The answer is ("  # {answer}."
        question = input + answer
        question = question.replace("  ", " ").strip()
        question_with_answer = question + f"{answer})"

        return question, question_with_answer, answer

    def get_question_text(self, problem):
        question = problem["question"]
        return question

    def get_context_text(self, problem, use_caption):
        txt_context = problem["hint"]
        img_context = problem["caption"] if use_caption else ""
        context = " ".join([txt_context, img_context]).strip()
        if context == "":
            context = "N/A"
        return context

    def get_choice_text(self, probelm, options=["A", "B", "C", "D", "E"]):
        choices = probelm["choices"]
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        return choice_txt

    def get_answer(self, problem, options=["A", "B", "C", "D", "E"]):
        return options[problem["answer"]]
