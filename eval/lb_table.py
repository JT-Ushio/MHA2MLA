#!/usr/bin/env python3
import json
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import pandas as pd


@dataclass
class Data:
    name: str
    path: Union[str, Path]
    # dim: Optional[int] = None  # noqa: F821
    task: str = "lbv1"
    length: str = ""
    data_path: Path = field(init=False)
    score: dict[str, float] = field(init=False)

    def __post_init__(self):
        if self.path == "":
            return
        if isinstance(self.path, str):
            self.path = Path(self.path)
        assert self.path.is_dir(), f"{self.path} is not a directory"
        if self.name == "":
            self.name = self.path.name
        if self.length == "":
            files = sorted(self.path.glob(f"{self.task}_*.jsonl"))
            assert files, f"No {self.task} files found in {self.path}"
            self.length = str(files[-1].stem[len(f"{self.task}_") :])
            if len(files) > 1:
                print(f"Warning: multiple {self.task} files found in {self.path}, using {self.length}")
        self.data_path = self.path / f"{self.task}_{self.length}.jsonl"
        assert self.data_path.is_file(), f"No {self.task} file found at {self.data_path}"

    def get_label(self):
        return f"{self.name}"
        # return f"{self.name} (test {self.length})"


# data = [
#     Data("", "saves/eval_pos/SmolLM-135M", ""),
#     Data("", "saves/eval_pos/SmolLM-360M", ""),
#     Data("", "saves/eval_pos/SmolLM-1.7B", ""),
#     Data("", "saves/eval_pos/SmolLM-1.7B-Instruct", ""),
#     Data("TinyLlama-1.1B", "saves/eval_pos/TinyLlama-1.1B-intermediate-step-1431k-3T", ""),
#     Data("", "saves/eval_pos/TinyLlama-NoPE-1.1B", ""),
# ]
data=[]

name_map = {
    "Single-Document QA": {
        "narrativeqa": "NarraQA",
        "qasper": "Qspr.",
        "multifieldqa_en": "MulF-en",  # TODO
        # "multifieldqa_zh": "MulF-zh"  # TODO
    },
    "Multi-Document QA": {
        "hotpotqa": "HpQA",
        "2wikimqa": "2WiMQA",
        "musique": "Musq.",
        # "dureader": "DuRd-zh"  # TODO
    },
    "Summarization": {
        "gov_report": "GRpt",
        "qmsum": "QMSum",
        "multi_news": "MultiN",
        # "vcsum": "VCS-zh"  # TODO
    },
    "Few-shot Learning": {
        "trec": "TREC",
        "triviaqa": "TrivQA",
        "samsum": "SAMSum",
        # "lsht": "LSHT-zh"  # TODO
    },
    "Synthetic Tasks": {
        "passage_count": "PsgC",
        "passage_retrieval_en": "PsgR-en",  # TODO
        # "passage_retrieval_zh": "PsgR-zh"  # TODO
    },
    "Code Completion": {
        "lcc": "LCC",
        "repobench-p": "RepoB-P",
    },
}


def parse_result(d: Data):
    score_list = defaultdict(list)
    with open(d.data_path, "r") as f:
        for line in f:
            raw = json.loads(line)
            score_list[raw["dataset"]].append(raw["score"])
    score = {k: sum(v) * 100 / len(v) for k, v in score_list.items()}
    d.score = score


def build_table():
    columns = [column for v in name_map.values() for column in v.keys()]
    df_columns = ["Model", "Length", "Avg."] + [column for v in name_map.values() for column in v.values()]
    # df_columns = [("", "Model"), ("", "Avg.")] + [
    #     (title, column) for title, v in name_map.items() for column in v.values()
    # ]
    # df_columns = pd.MultiIndex.from_tuples(df_columns)
    df_data = []
    for d in data:
        row = [d.score[column] for column in columns]
        row = [d.get_label(), d.length, sum(row) / len(row)] + row
        df_data.append(row)
    df = pd.DataFrame(df_data, columns=df_columns)
    print(df)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    path = Path(parser.parse_args().path)
    dirs = sorted([x for x in path.iterdir() if x.is_dir()])
    for d in dirs:
        data.append(Data("", d))
    for d in data:
        parse_result(d)
    df = build_table()
    with open("longbench.md", "w") as f:
        f.write(df.round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
