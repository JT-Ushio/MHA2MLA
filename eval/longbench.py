import bisect
from dataclasses import dataclass
import os
import re
import string
from collections import Counter, defaultdict
from itertools import accumulate
from typing import TYPE_CHECKING, Optional
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
import logging
import jieba
import pandas as pd
import torch#
import transformers
from accelerate.utils import gather_object
from datasets import load_dataset
from fuzzywuzzy import fuzz
from rouge import Rouge
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrainerCallback
from typing_extensions import override
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


@dataclass
class LongBenchArguments:
    longbench: bool = False
    lb_max_tokens: int = 2048
    lb_batch_size: int = 16


@dataclass
class MLAArguments:
    is_partial_rope: bool = False
    is_mla: bool = False
    model_path: str = ""
    tokenizer_path: str = ""
    dtype: str = "float32"


@dataclass
class CacheArguments:
    cache_implementation: str = None
    backend: str = "quanto"
    nbits: int = 8
    residual_length: int = 128



logger = logging.getLogger(__name__)

LEN2NAME = {1 << (10 + i): f"{1<<i}k" for i in range(9)}


class RunLongBenchCallback(TrainerCallback):
    def __init__(
        self,
        args: "LongBenchArguments",
        trainer: "Trainer",
        cache_args: "CacheArguments",
    ):
        super().__init__()
        self.args = args
        self.cache_args = cache_args
        self.trainer = trainer
        self.model = trainer.model
        self.tokenizer: "PreTrainedTokenizerBase" = trainer.tokenizer  # type: ignore
        self.tokenizer.padding_side = "left"
        self.is_world_process_zero = self.trainer.is_world_process_zero()

        dataset = LongBenchV1Dataset(args, self.tokenizer)
        dataloader = DataLoader(
            dataset, batch_size=self.args.lb_batch_size, collate_fn=dataset.collate_fn
        )
        self.dataloaderV1 = self.trainer.accelerator.prepare(dataloader)

    @override
    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        output_dir = None
        if not args.do_train:
            output_dir = args.output_dir  # save result when only do_eval
        self.run(output_dir)

    @torch.no_grad()
    def run(self, output_dir: Optional[str] = None):
        self.runV1(self.dataloaderV1, "lbv1", output_dir)


    def runV1(
        self,
        dataloader,
        task: str,
        output_dir: Optional[str] = None,
        soft_eval: bool = False,
    ):
        with torch.cuda.device(self.model.device):
            torch.cuda.empty_cache()
        if output_dir is not None and not os.path.exists(output_dir):
            output_dir = None
            logger.warning(f"{task}: {output_dir=} does not exist, skipping output.")
        self.model.eval()
        rst = []
        for batch in tqdm(dataloader, disable=not self.is_world_process_zero):
            _id = batch.pop("_id")
            dataset = batch.pop("dataset")
            answers = batch.pop("answers")
            all_classes = batch.pop("all_classes")
            max_gen = batch.pop("max_gen")
            should_save = batch.pop("save")
            pred = self.inferV1(batch, max_gen)
            score = [
                self.evalV1(d, a, c, p)
                for d, a, c, p in zip(dataset, answers, all_classes, pred)
            ]
            assert (
                len(_id)
                == len(dataset)
                == len(answers)
                == len(all_classes)
                == len(max_gen)
                == len(should_save)
                == len(pred)
                == len(score)
            ), f"{len(_id)=}, {len(dataset)=}, {len(answers)=}, {len(all_classes)=}, {len(max_gen)=}, {len(should_save)=} {len(pred)=}, {len(score)=}"
            for i, d, a, c, save, p, s in zip(
                _id, dataset, answers, all_classes, should_save, pred, score
            ):
                rst.append((i, s, d, p if save else "", a, c))
        rst = gather_object(rst)
        rst_bak = rst
        rst = []
        rst_keys = {}
        for _id, *rest in rst_bak:
            if _id not in rst_keys:
                rst_keys[_id] = len(rst)
                rst.append((_id, *rest))
        logger.warning_once(f"{task} all gather {len(rst)=}")
        assert len(rst) == len(dataloader.dataset)

        if self.is_world_process_zero:
            df = pd.DataFrame(
                rst,
                columns=["_id", "score", "dataset", "pred", "answers", "all_classes"],
            )
            name = LEN2NAME[self.args.lb_max_tokens]
            if output_dir is not None:
                # run_name = (
                #     f"_{self.trainer.args.run_name}"
                #     if self.trainer.args.run_name
                #     else ""
                # )
                run_name = ""
                file_path = os.path.join(output_dir, f"{task}{run_name}_{name}.jsonl")

                parent_dir = os.path.dirname(file_path)
                os.makedirs(parent_dir, exist_ok=True)

                if os.path.exists(file_path):
                    logger.warning(f"{file_path} already exists, overwriting.")
                else:
                    logger.info(f"Saving {task} to {file_path}")
                df.to_json(file_path, orient="records", lines=True, force_ascii=True)
            group_mean = df.groupby("dataset")["score"].mean()
            overall_mean = group_mean.mean()
            logger.info(f"{task} dataset score: {group_mean}")
            logger.info(f"{task} overall score: {overall_mean * 100:.1f}")

    def inferV1(self, batch, max_gen: list[int]):
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = max(max_gen)
        gen_config.num_beams = 1
        gen_config.do_sample = False
        gen_config.pad_token_id = self.tokenizer.eos_token_id
        cache_config = {
            "backend": self.cache_args.backend,
            "nbits": self.cache_args.nbits,
            "residual_length": self.cache_args.residual_length,
            "compute_dtype": self.model.dtype,
            "device": self.model.device,
        }
        output = self.model.generate(
            **batch,
            generation_config=gen_config,
            cache_implementation=self.cache_args.cache_implementation,
            **(
                {"cache_config": cache_config}
                if self.cache_args.cache_implementation == "quantized"
                else {}
            ),
        )
        pred = [
            self.tokenizer.decode(output[i][-gen_config.max_new_tokens :][:gen_len])
            for i, gen_len in enumerate(max_gen)
        ]
        return pred

    def evalV1(
        self, dataset: str, answers: list[str], all_classes: list[str], pred: str
    ):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip("\n").split("\n")[0]
        for ground_truth in answers:
            score = max(
                score,
                dataset2metric[dataset](pred, ground_truth, all_classes=all_classes),
            )
        return score

    def _get_tb_writer(self):
        for callback in self.trainer.callback_handler.callbacks:
            if isinstance(callback, transformers.integrations.TensorBoardCallback):
                return callback.tb_writer
        return None

    def tb_add_scalar(self, *args, **kwargs):
        if self.tb_writer is None:
            self.tb_writer = self._get_tb_writer()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(*args, **kwargs)


class LongBenchV2Dataset(Dataset):
    prompt_template = """
Please read the following text and answer the question below.
<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

The correct answer is
"""

    def __init__(
        self, args: "LongBenchArguments", tokenizer: "PreTrainedTokenizerBase"
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = load_dataset("data/LongBench-v2", split="train")

    def __getitem__(self, idx):
        data = self.dataset[idx]  # type: ignore
        assert isinstance(data, dict)
        context_max = 8 * self.args.lb_max_tokens  # text_len/token_len <4 for en
        context_half = context_max // 2
        prompt_truncated = False

        if len(data["context"]) > context_max:  # truncate to save CPU
            data["context"] = (
                data["context"][:context_half] + data["context"][-context_half:]
            )
            prompt_truncated = True
        prompt = self.prompt_template.strip().format(**data)  # strip to remove extra \n
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_max = self.args.lb_max_tokens - 1  # for answer token
        half = prompt_max // 2
        if input_ids.shape[0] > prompt_max:
            input_ids = torch.cat([input_ids[: prompt_max - half], input_ids[-half:]])
            assert input_ids.shape[0] == prompt_max
        # else:
        #     assert not prompt_truncated
        return {
            "input_ids": input_ids,
            "_id": data["_id"],
            "difficulty": data["difficulty"],
            "length": data["length"],
            "answer": data["answer"],
        }

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def collate_fn(self, samples):
        batch = {}
        for k in samples[0].keys():
            batch[k] = [f[k] for f in samples]
        batch.update(
            self.tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
        )
        return batch


class LongBenchV1Dataset(Dataset):
    dataset2prompt = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
        "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
        "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    }
    dataset2maxlen = {
        "narrativeqa": 128,
        "qasper": 128,
        "multifieldqa_en": 64,
        "multifieldqa_zh": 64,
        "hotpotqa": 32,
        "2wikimqa": 32,
        "musique": 32,
        "dureader": 128,
        "gov_report": 512,
        "qmsum": 512,
        "multi_news": 512,
        "vcsum": 512,
        "trec": 64,
        "triviaqa": 32,
        "samsum": 128,
        "lsht": 64,
        "passage_count": 32,
        "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32,
        "lcc": 64,
        "repobench-p": 64,
    }

    def __init__(
        self, args: "LongBenchArguments", tokenizer: "PreTrainedTokenizerBase"
    ):
        self.args = args
        self.tokenizer = tokenizer
        # fmt: off
        subsets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # fmt: on
        self.datasets = [
            load_dataset(
                "THUDM/LongBench",
                subset,
                split="test",
                trust_remote_code=True,
            )
            for subset in subsets
        ]
        self.acc_lens = [0] + list(
            accumulate([len(dataset) for dataset in self.datasets])
        )
        self.length = self.acc_lens[-1]

    def __getitem__(self, idx):
        # find data using acc_lens
        dataset_idx = bisect.bisect_right(self.acc_lens, idx) - 1
        data = self.datasets[dataset_idx][idx - self.acc_lens[dataset_idx]]
        assert isinstance(data, dict)
        name: str = data["dataset"]
        prompt_template = self.dataset2prompt[name]
        context_max = 8 * self.args.lb_max_tokens  # text_len/token_len <4 for en
        context_half = context_max // 2
        prompt_truncated = False
        if len(data["context"]) > context_max:  # truncate to save CPU
            data["context"] = (
                data["context"][:context_half] + data["context"][-context_half:]
            )
            prompt_truncated = True
        prompt = prompt_template.format(**data)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_max = (
            self.args.lb_max_tokens - self.dataset2maxlen[name]
        )  # for answer token
        half = prompt_max // 2
        if input_ids.shape[0] > prompt_max:
            input_ids = torch.cat([input_ids[: prompt_max - half], input_ids[-half:]])
            assert input_ids.shape[0] == prompt_max
        else:
            assert not prompt_truncated
        return {
            "input_ids": input_ids,
            "_id": data["_id"],
            "dataset": name,
            "answers": data["answers"],
            "all_classes": data["all_classes"],
            "max_gen": self.dataset2maxlen[name],
            "save": idx - self.acc_lens[dataset_idx] < 10,
        }

    def __len__(self):
        return self.length

    def collate_fn(self, samples):
        batch = {}
        for k in samples[0].keys():
            batch[k] = [f[k] for f in samples]
        batch.update(
            self.tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
        )
        return batch


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = HfArgumentParser(
        (TrainingArguments, LongBenchArguments, MLAArguments, CacheArguments)
    )
    training_args, lb_args, mla_args, cache_args = parser.parse_args_into_dataclasses()
    model_path = mla_args.model_path
    tokenizer_path = (
        mla_args.tokenizer_path
        if mla_args.tokenizer_path is not None
        else mla_args.model_path
    )
    if mla_args.is_partial_rope:
        raise NotImplementedError("Not implemented for partial-rope")
    if mla_args.is_mla:
        import json, os

        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        from monkey_patch import mla_monkey_patch
        mla_monkey_patch(config["RoPE"])

    if mla_args.dtype == "float32":
        dtype = torch.float32
    elif mla_args.dtype == "float16":
        dtype = torch.float16
    elif mla_args.dtype == "bfloat16":
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    longbench_callback = RunLongBenchCallback(
        lb_args, trainer=trainer, cache_args=cache_args
    )
    # trainer.callback_handler.callbacks.append(longbench_callback)
    trainer.add_callback(longbench_callback)

    trainer.callback_handler.on_evaluate(
        trainer.args, trainer.state, trainer.control, None
    )


if __name__ == "__main__":
    main()
