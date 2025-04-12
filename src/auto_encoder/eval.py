import argparse
import os
from dataclasses import asdict
from pprint import pformat

from lighteval.parsers import (
    parser_accelerate,
    parser_baseline,
    parser_nanotron,
    parser_utils_tasks,
)
from lighteval.tasks.registry import Registry, taskinfo_selector
import yaml


CACHE_DIR = os.getenv("HF_HOME")


# copied from lighteval.main
def cli_evaluate():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="CLI tool for lighteval, a lightweight framework for LLM evaluation"
    )
    parser.add_argument(
        "--is_auto_encoder",
        action="store_true",
        help="whether to use the auto encoder model",
    )

    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # Subparser for the "accelerate" command
    parser_a = subparsers.add_parser(
        "accelerate", help="use accelerate and transformers as backend for evaluation."
    )
    parser_accelerate(parser_a)

    # Subparser for the "nanotron" command
    parser_b = subparsers.add_parser(
        "nanotron", help="use nanotron as backend for evaluation."
    )
    parser_nanotron(parser_b)

    parser_c = subparsers.add_parser("baseline", help="compute baseline for a task")
    parser_baseline(parser_c)

    # Subparser for task utils functions
    parser_d = subparsers.add_parser(
        "tasks", help="display information about available tasks and samples."
    )
    parser_utils_tasks(parser_d)

    args = parser.parse_args()

    # Monkey patching for the partial rope
    if args.is_auto_encoder:
        from monkey_patch import ae_patch_func_hf

        ae_patch_func_hf()

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as main_accelerate

        main_accelerate(args)


if __name__ == "__main__":
    cli_evaluate()
