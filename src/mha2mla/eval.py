import argparse
import os
from dataclasses import asdict
from pprint import pformat

from lighteval.parsers import parser_accelerate, parser_baseline, parser_nanotron, parser_utils_tasks
from lighteval.tasks.registry import Registry, taskinfo_selector
import yaml,json


CACHE_DIR = os.getenv("HF_HOME")


# copied from lighteval.main
def cli_evaluate():  # noqa: C901
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    parser.add_argument(
        "--is_partial_rope",
        action="store_true",
        help="Path to the RoPE configuration file. This file should contain the partial-RoPE configuration.",
    )
    parser.add_argument(
        "--is_mla",
        action="store_true",
        help="Whether the model is a MLA model. If True, the model will be evaluated using the MLA architecture.",
    )

    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # Subparser for the "accelerate" command
    parser_a = subparsers.add_parser("accelerate", help="use accelerate and transformers as backend for evaluation.")
    parser_accelerate(parser_a)

    args = parser.parse_args()
    if args.is_mla or args.is_partial_rope:
        from monkey_patch import partial_rope_monkey_patch, mla_monkey_patch
        model_args = args.model_args
        model_args = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in model_args.split(",")}
        ckpt_path = model_args["pretrained"]
        with open(os.path.join(ckpt_path,"config.json")) as f:
            model_config = json.load(f)
        cfg_RoPE = model_config["RoPE"]
        assert not (args.is_mla and args.is_partial_rope), "Cannot set both is_mla and is_partial_rope to True."
        if args.is_mla:
            mla_monkey_patch(cfg_RoPE)
        else:
            partial_rope_monkey_patch(cfg_RoPE)

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as main_accelerate

        main_accelerate(args)
    else:
        print("You need to set the subcommand to 'accelerate'.")


if __name__ == "__main__":
    cli_evaluate()
