import re
import numpy as np

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics, SampleLevelMetric, MetricCategory, MetricUseCase, ExactMatches
import lighteval.tasks.default_prompts as prompt
from .math_utils import math_normalizer


def prompt_hellaswag(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
    )

def prompt_commonsense_qa(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"].strip()),
        instruction="",
    )

def mmlu_pro_mc_prompt(line, task_name: str = None):
    options = line["options"]
    letters = [chr(ord("A") + i) for i in range(len(options))]
    topic = line["category"].replace('_', ' ')
    query = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{letter}. {choice}\n" for letter, choice in zip(letters, options)])
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=letters,
        gold_index=line["answer_index"],
        instruction=f"The following are multiple choice questions (with answers) about {topic}.\n\n",
    )

def bbh_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="Question: " + line["input"] + "\nAnswer: ",
        choices=[line["target"]],
        gold_index=0,
    )

def prompt_math(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Problem:\n{line['problem']}\n\nSolution:\n",
        gold_index=0,
        choices=[f"{line['solution']}\n"],
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function=prompt.arc,
        suite=["custom"],
        hf_repo="ai2_arc",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        hf_subset=f"ARC-Easy",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function=prompt.arc,
        suite=["custom"],
        hf_repo="ai2_arc",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        hf_subset=f"ARC-Challenge",
        evaluation_splits=["test"],
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="openbook_qa",
        prompt_function=prompt.openbookqa,
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function=prompt_hellaswag,
        suite=["custom"],
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function=prompt_commonsense_qa,
        suite=["custom"],
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function=prompt.winogrande,
        suite=["custom"],
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function=prompt.piqa_harness,
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_subset="plain_text",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        trust_dataset=True,
        metric=[Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=prompt.triviaqa,
        suite=["custom"],
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metric=[Metrics.quasi_exact_match_triviaqa],
        generation_size=20,
        trust_dataset=True,
        stop_sequence=["\n", ".", ","],
        # stop_sequence=["Question:", "Question"],
        few_shots_select="random_sampling_from_train",
    ),
    LightevalTaskConfig(
        name="mmlu_pro",
        prompt_function=mmlu_pro_mc_prompt,
        suite=["custom"],
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        metric=[Metrics.loglikelihood_acc_norm_nospace],
        evaluation_splits=["test"],
        few_shots_split="validation",
    ),
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,
        suite=["custom"],
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.quasi_exact_match_gsm8k],
        generation_size=256,
        stop_sequence=["Question:", "Question"],
        few_shots_select="random_sampling_from_train",
    )
]

BBH_TASKS = [
    LightevalTaskConfig(
        name=f"bbh:{subset}",
        prompt_function=bbh_prompt,
        suite=["custom"],
        hf_repo="lighteval/big_bench_hard",
        hf_subset=subset,
        hf_revision="80610173426f05e6f1448f047e2db4840a7dd899",
        metric=[Metrics.exact_match],
        hf_avail_splits=["train"],
        # this is the only split available, obviously not used in training
        evaluation_splits=["train"],
        few_shots_split="train",
        trust_dataset=True,
        stop_sequence=["Question:", "Question"],
    )
    for subset in [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]
]

TASKS_TABLE.extend(BBH_TASKS)

quasi_exact_match_math = SampleLevelMetric(
        metric_name="qem",
        sample_level_fn=ExactMatches(
            strip_strings=True, normalize_pred=math_normalizer, normalize_gold=math_normalizer
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.MATH,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

MATH_TASKS = [
    LightevalTaskConfig(
        name=f"math:{subset}",
        prompt_function=prompt_math,
        suite=["custom"],
        hf_repo="HuggingFaceTB/MATH",
        hf_subset=subset,
        hf_revision="140a673f1f7182daf7923fdc7108e8cdbf97df46",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[quasi_exact_match_math],
        generation_size=1024,
        stop_sequence=["Problem:"],
        few_shots_split="fewshot",
        few_shots_select="sequential",
        trust_dataset=True,
    )
    for subset in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]

TASKS_TABLE.extend(MATH_TASKS)


if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))