import argparse
import os, sys

from transformers.modeling_utils import load_sharded_checkpoint

from lighteval.models.utils import _get_dtype, _simplify_name, batched
from lighteval.pipeline import (
    EnvConfig,
)
import transformers
from lighteval.parsers import (
    parser_accelerate,
)
from lighteval.models.model_config import BaseModelConfig
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)
import types
from lighteval.models.model_loader import BaseModel

current_file_path = os.path.abspath(__file__)
target_directory = os.path.join(
    os.path.dirname(os.path.dirname(current_file_path)), "src", "mha2mla"
)
sys.path.append(str(target_directory))
from patching_model_load import patch_model
from patching_qwen2 import mha2mla_qwen2
from patching_llama import mha2mla_llama

from safetensors.torch import load_file


def create_load_func(mha2mla_args):

    def _create_auto_model(
        self, config: BaseModelConfig, env_config: EnvConfig
    ) -> transformers.PreTrainedModel:
        config.model_parallel, max_memory, device_map = self.init_model_parallel(
            config.model_parallel
        )
        torch_dtype = _get_dtype(config.dtype, self._config)
        ckpt_path = config.pretrained
        if mha2mla_args is None or mha2mla_args.is_baseline:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                max_memory=max_memory,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=config.trust_remote_code,
                cache_dir=env_config.cache_dir,
                offload_folder=env_config.cache_dir,
                token=env_config.token,
                quantization_config=config.quantization_config,
            )
        else:
            # Instance
            model_config = AutoConfig.from_pretrained(ckpt_path)
            mha_model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=config.trust_remote_code,
            )
            mla_model, q_idx, k_idx = patch_model(mha_model, model_config, mha2mla_args)
            if isinstance(mha_model, LlamaForCausalLM):
                mha2mla_llama(q_idx, k_idx)
            elif isinstance(mha_model, Qwen2ForCausalLM):
                mha2mla_qwen2(q_idx, k_idx)
            # Load weights
            signle_weight_file = os.path.join(ckpt_path, "model.safetensors")
            if os.path.exists(signle_weight_file):
                state_dict = load_file(signle_weight_file)
                mla_model.load_state_dict(state_dict)
            else:
                load_result = load_sharded_checkpoint(mla_model, ckpt_path)
            model = mla_model.to(dtype=torch_dtype)
        return model
    return _create_auto_model


# copied from lighteval.main
def cli_evaluate():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="CLI tool for lighteval, a lightweight framework for LLM evaluation"
    )

    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # Subparser for the "accelerate" command
    parser_a = subparsers.add_parser(
        "accelerate", help="use accelerate and transformers as backend for evaluation."
    )
    parser_accelerate(parser_a)

    args = parser.parse_args()
    model_args = args.model_args
    model_args = {
        k.split("=")[0]: k.split("=")[1] if "=" in k else True
        for k in model_args.split(",")
    }
    ckpt_path = model_args["pretrained"]
    model_config = AutoConfig.from_pretrained(ckpt_path)

    if (
        hasattr(model_config, "mha2mla_args")
        and not model_config.mha2mla_args["is_baseline"]
    ):
        mha2mla_args = types.SimpleNamespace(**model_config.mha2mla_args)
        BaseModel._create_auto_model = create_load_func(mha2mla_args)

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as original_main_accelerate

        # main_accelerate(args, model, ckpt_path)
        original_main_accelerate(args)
    else:
        print("You need to set the subcommand to 'accelerate'.")


if __name__ == "__main__":
    cli_evaluate()
