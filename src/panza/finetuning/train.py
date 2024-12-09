# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import glob
import logging
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import spops

import torch
from composer import Trainer
from composer.core.callback import Callback
from composer.metrics.nlp import (
    InContextLearningCodeEvalAccuracy,
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
    LanguageCrossEntropy,
    LanguagePerplexity,
)
from composer.optim import DecoupledAdamW
from composer.profiler import JSONTraceHandler, Profiler, TraceHandler, cyclic_schedule
from composer.utils import dist, get_device, reproducibility
from datasets import disable_caching
from llmfoundry import ComposerHFCausalLM
from llmfoundry.eval.metrics.nlp import InContextLearningMetric
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithFSDP
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils import find_mosaicml_logger, log_train_analytics, maybe_create_mosaicml_logger
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from peft import get_peft_model
from peft.tuners.rosa import RosaConfig, RosaModel, RosaScheduler
from rich.traceback import install
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase

install()
from llmfoundry.callbacks import AsyncEval
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.utils.builders import (
    add_metrics_to_eval_loaders,
    build_algorithm,
    build_callback,
    build_evaluators,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    log_config,
    pop_config,
    process_init_device,
    update_batch_size_info,
)
from llmfoundry.utils.registry_utils import import_file

import hydra
from omegaconf import DictConfig, OmegaConf

from panza import PanzaWriter  # The import also loads custom Hydra resolvers

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if "eval_loader" in cfg:
        eval_loader = cfg.eval_loader
        if isinstance(eval_loader, ListConfig):
            for loader in eval_loader:
                if loader.label is None:
                    raise ValueError(
                        "When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute."
                    )
                loaders.append(loader)
        else:
            loaders.append(eval_loader)
    for loader in loaders:
        if loader.name == "text":
            if cfg.model.name == "hf_t5":
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " '
                    + f"dataloader. Only finetuning is supported."
                )

    if "icl_tasks" in cfg:
        if cfg.model.name == "hf_t5":
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (
        cfg.model.get("fc_type", "torch") != "te"
        and "te" not in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp")
        and "fp8" in cfg.precision
    ):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            + "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if cfg.model.get("fc_type", "torch") == "te" or "te" in cfg.model.get("ffn_config", {}).get(
        "ffn_type", "mptmlp"
    ):
        fsdp_config = cfg.get("fsdp_config", None)
        act_ckpt = fsdp_config.get("activation_checkpointing", False)
        act_ckpt_reentrant = fsdp_config.get("activation_checkpointing_reentrant", False)
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == True:
            warnings.warn(
                "`te.Linear` layers do not support activation_checkpointing with "
                + "`activation_checkpointing_reentrant = True`. "
                + "Setting cfg.fsdp_config.activation_checkpointing_reentrant=False."
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = False

    if cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp") == "te_ln_mlp":
        warnings.warn(
            "`te.LayerNormMLP` requires has issues with torch._dynamo. "
            + "Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager."
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    if cfg.model.get("load_in_8bit", False):
        raise ValueError("`load_in_8bit` is only supported for evaluation rather than training.")

    if cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp") in ffns_with_megablocks:
        moe_world_size = cfg.model.get("ffn_config", {}).get("moe_world_size", 1)
        use_orig_params = cfg.get("fsdp_config", {}).get("use_orig_params", True)
        if moe_world_size > 1 and not use_orig_params:
            raise ValueError(
                f"MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`."
            )


def create_run_name(cfg: DictConfig) -> str:
    # export RUN_NAME=panza_${PANZA_USERNAME}_${MODEL_TYPE}_${MODEL_PRECISION}-bs${BS}-fft-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}${PREAMBLE_STR}${RAFT_STR}

    run_name = f"panza_{cfg.user.username}"

    model_name = cfg.finetuning.model_name_or_path.split("/")[-1]
    run_name += f"-{model_name}"

    run_name += f"-{cfg.model_precision}"
    run_name += f"-bs{cfg.finetuning.batch_size}"

    if hasattr(cfg.finetuning, "rosa"):
        run_name += "-rosa"
    else:
        run_name += "-fft"

    run_name += f"-lr{cfg.finetuning.lr}"
    run_name += f"-{cfg.finetuning.max_duration}"
    run_name += f"-seed{cfg.finetuning.seed}"

    return run_name


def override_rosa_schedule(cfg: DictConfig, mask_generation=False) -> None:
    # Disable struct mode to allow modifications
    rosa_cfg = cfg.finetuning.rosa
    OmegaConf.set_struct(rosa_cfg, False)

    mask_path = str(Path(cfg.checkpoint_dir) / "masks" / cfg.finetuning.run_name)

    if mask_generation:
        rosa_cfg.schedule = "wl16" if rosa_cfg.lora_r != 0 else "spa_only"
        rosa_cfg.mask_load_path = None
        rosa_cfg.mask_save_path = mask_path
        rosa_cfg.terminate_after_mask_generation = True
        rosa_cfg.mask_gen_model_precision = "amp_bf16"
    else:
        if rosa_cfg.spa_d > 0 and rosa_cfg.lora_r != 0:
            rosa_cfg.schedule = "default"
        elif rosa_cfg.lora_r != 0:
            rosa_cfg.schedule = "lora_only"
            rosa_cfg.mask_load_path = None
        else:
            rosa_cfg.schedule = "spa_only"

        rosa_cfg.mask_load_path = mask_path
        rosa_cfg.mask_save_path = None
        rosa_cfg.terminate_after_mask_generation = None

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(rosa_cfg, True)


def create_checkpoint_dirs(cfg: DictConfig) -> None:
    # Create model directory
    os.makedirs(os.path.join(cfg.checkpoint_dir, "models"), exist_ok=True)

    # Create mask directory
    if hasattr(cfg.finetuning, "rosa"):
        os.makedirs(os.path.join(cfg.checkpoint_dir, "masks"), exist_ok=True)


def get_hf_save_precision(cfg: DictConfig) -> str:
    if cfg.model_precision == "bf16":
        return "bfloat16"
    elif cfg.model_precision == "fp32":
        return "float32"
    else:
        raise ValueError(f"Unsupported model_precision: {cfg.model_precision}")


def get_rosa_dtype(cfg: DictConfig) -> str:
    if cfg.model_precision == "bf16":
        return "bf16"
    elif cfg.model_precision == "fp32":
        return "fp32"
    elif cfg.model_precision == "4bit":
        return "fp32"
    else:
        raise ValueError(f"Unsupported model_precision: {cfg.model_precision}")


def override_config(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    if not cfg.finetuning.run_name:
        cfg.finetuning.run_name = create_run_name(cfg)

    if hasattr(cfg.finetuning, "rosa"):
        cfg.finetuning.rosa.rosa_dtype = get_rosa_dtype(cfg)
        if cfg.finetuning.rosa.spa_d != 0:
            override_rosa_schedule(cfg, mask_generation=cfg.finetuning.rosa.masks_only)
    else:
        cfg.finetuning.callbacks.hf_checkpointer.precision = get_hf_save_precision(cfg)

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)


def save_config_to_yaml(cfg: DictConfig) -> str:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as temp_file:
        OmegaConf.save(config=cfg, f=temp_file.name)
        return temp_file.name


def build_composer_peft_model(
    model_config: str,
    rosa_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    is_fsdp: bool = False,
) -> ComposerHFCausalLM:

    # 1) loads a hf model, 2) adds peft modules, 3) wraps it in a ComposerHFCausalLM.
    print("Building model from HuggingFace checkpoint...")

    weight_bias_dtype = model_config.get("weight_bias_dtype", None)
    if weight_bias_dtype == "4bit":
        compute_dtype = torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif weight_bias_dtype == "bf16":
        compute_dtype = torch.bfloat16
        quant_config = None
    else:
        assert weight_bias_dtype == "fp32"
        compute_dtype = torch.float32
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.pretrained_model_name_or_path,
        device_map="cpu" if quant_config is None else "auto",
        torch_dtype=compute_dtype,
        # load_in_4bit=weight_bias_dtype == '4bit',
        quantization_config=quant_config,
        trust_remote_code=True,
        use_auth_token=True,
        use_cache=False,
        attn_implementation="eager",
    )

    print("Model built!")
    if rosa_config is not None:
        print("Building RoSA config...")
        config = RosaConfig(
            r=rosa_config["lora_r"],
            d=rosa_config["spa_d"],
            lora_alpha=rosa_config.get("lora_alpha", 16),
            target_modules=rosa_config.get("target_modules", "all-linear"),
            lora_dropout=rosa_config.get("lora_dropout", 0.05),
            impl=rosa_config.get("impl", "auto"),
            spa_store_transpose=rosa_config.get("spa_store_transpose", True),
            rosa_dtype=rosa_config.get("rosa_dtype", True),
            spa_num_grads=rosa_config.get("spa_num_grads", 1),
            grad_acc_mode=rosa_config.get("grad_acc_mode", "mean_squared"),
            grad_4bit_accum=rosa_config.get("grad_4bit_accum", False),
            mask_load_path=rosa_config.get("mask_load_path", None),
            mask_save_path=rosa_config.get("mask_save_path", None),
            terminate_after_mask_generation=rosa_config.get(
                "terminate_after_mask_generation", False
            ),
            schedule=rosa_config.get("schedule", "df"),
            bias="none",
            task_type="CAUSAL_LM",
        )
        # raise ValueError(config)
        print("Adding RoSA modules...")
        model = get_peft_model(model, config)
        print("RoSA modules added!")

    train_metrics = [LanguageCrossEntropy(), LanguagePerplexity()]
    eval_metrics = [
        LanguageCrossEntropy(),
        LanguagePerplexity(),
        InContextLearningLMAccuracy(),
        InContextLearningMultipleChoiceAccuracy(),
        InContextLearningQAAccuracy(),
        InContextLearningCodeEvalAccuracy(),
        InContextLearningLMExpectedCalibrationError(),
        InContextLearningMCExpectedCalibrationError(),
    ]

    model = HuggingFaceModelWithFSDP(
        model=model,
        shift_labels=True,
        tokenizer=tokenizer,
        metrics=train_metrics,
        eval_metrics=eval_metrics,
        init_device="cpu",
        peft_config=None,
    )
    return model


@hydra.main(version_base="1.1", config_path="../../../configs", config_name="panza_finetuning")
def main(cfg: DictConfig) -> Trainer:
    override_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # The preprocessing config is saved to a temporary directory
    # and accessed through an environment variable. Note that this
    # happens separately for each process (however, a collision should)
    # not be a problem, since the configs are the same.
    OmegaConf.set_struct(cfg, False)
    cfg.preprocessing.model = cfg.finetuning.model_name_or_path
    preprocessing_yaml = save_config_to_yaml(cfg.preprocessing)

    environment = os.environ
    environment["WANDB_PROJECT"] = f"panza-{cfg.user.username}"
    environment["WANDB_DISABLED"] = str(int(cfg.finetuning.wandb_disabled))
    environment["PANZA_PREPROCESSING_CONFIG"] = preprocessing_yaml

    cfg = cfg.finetuning

    # Make the config editable for popping.
    OmegaConf.set_struct(cfg, False)

    # Run user provided code if specified
    code_paths = pop_config(cfg, "code_paths", must_exist=False, default_value=[], convert=True)
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)

    cuda_alloc_conf = []
    # Get max split size mb
    max_split_size_mb: Optional[int] = cfg.pop("max_split_size_mb", None)
    if max_split_size_mb is not None:
        cuda_alloc_conf.append(f"max_split_size_mb:{max_split_size_mb}")

    # Expandable segments
    if cfg.pop("expandable_segments", False):
        cuda_alloc_conf.append("expandable_segments:True")

    if len(cuda_alloc_conf) > 0:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(cuda_alloc_conf)

    # Set CUDA lazy loading
    # This can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = cfg.pop("cuda_load_lazy", False)
    if cuda_load_lazy:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    # Set seed first
    seed: int = pop_config(cfg, "seed", must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(
        cfg, "dist_timeout", must_exist=False, default_value=600.0
    )
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    save_merged_model: bool = pop_config(cfg, "save_merged_model", False)

    # Get global and device batch size information from distributed/single node setting
    cfg = update_batch_size_info(cfg)
    logged_cfg.update(cfg, merge=True)

    # Mandatory model training configs
    model_config: DictConfig = pop_config(cfg, "model", must_exist=True)
    tokenizer_config: Dict[str, Any] = pop_config(cfg, "tokenizer", must_exist=True, convert=True)
    optimizer_config: Dict[str, Any] = pop_config(cfg, "optimizer", must_exist=True, convert=True)
    scheduler_config: Dict[str, Any] = pop_config(cfg, "scheduler", must_exist=True, convert=True)
    train_loader_config: DictConfig = pop_config(cfg, "train_loader", must_exist=True)

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "fsdp_config", must_exist=False, default_value=None, convert=True
    )

    ds_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "ds_config", must_exist=False, default_value=None, convert=True
    )

    rosa_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "rosa", must_exist=False, default_value=None, convert=True
    )

    hf_save_path: Union[int, str] = pop_config(cfg, "hf_save_path", must_exist=True)

    eval_loader_config: Optional[Union[DictConfig, ListConfig]] = pop_config(
        cfg, "eval_loader", must_exist=False, default_value=None
    )
    icl_tasks_config: Optional[Union[ListConfig, str]] = pop_config(
        cfg, "icl_tasks", must_exist=False, default_value=None
    )
    eval_gauntlet_config: Optional[Union[DictConfig, str]] = pop_config(
        cfg, "eval_gauntlet", must_exist=False, default_value=None
    )
    icl_subset_num_batches: Optional[int] = pop_config(
        cfg, "icl_subset_num_batches", must_exist=False, default_value=None
    )
    icl_seq_len: Optional[int] = pop_config(
        cfg, "icl_seq_len", must_exist=False, default_value=None
    )
    # Optional logging, evaluation and callback configs
    logger_configs: Optional[DictConfig] = pop_config(
        cfg, "loggers", must_exist=False, default_value=None, convert=True
    )
    callback_configs: Optional[DictConfig] = pop_config(
        cfg, "callbacks", must_exist=False, default_value=None, convert=True
    )
    algorithm_configs: Optional[DictConfig] = pop_config(
        cfg, "algorithms", must_exist=False, default_value=None
    )

    # Mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(cfg, "device_train_batch_size", must_exist=True)
    device_eval_batch_size: int = pop_config(cfg, "device_eval_batch_size", must_exist=True)
    max_duration: Union[int, str] = pop_config(cfg, "max_duration", must_exist=True)
    eval_interval: Union[int, str] = pop_config(
        cfg, "eval_interval", default_value=1, must_exist=False
    )
    precision: str = pop_config(cfg, "precision", must_exist=True)
    max_seq_len: int = pop_config(cfg, "max_seq_len", must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get("RUN_NAME", "llm")
    run_name: str = pop_config(cfg, "run_name", must_exist=False, default_value=default_run_name)
    save_folder: Optional[str] = pop_config(
        cfg, "save_folder", must_exist=False, default_value=None
    )
    is_state_dict_sharded: bool = (
        (fsdp_config.get("state_dict_type", "full") == "sharded") if fsdp_config else False
    )
    save_latest_filename: str = pop_config(
        cfg,
        "save_latest_filename",
        must_exist=False,
        default_value=(
            "latest-sharded-rank{rank}" if is_state_dict_sharded else "latest-rank{rank}.pt"
        ),
    )
    save_overwrite: bool = pop_config(cfg, "save_overwrite", must_exist=False, default_value=False)
    save_weights_only: bool = pop_config(
        cfg, "save_weights_only", must_exist=False, default_value=False
    )
    save_filename: str = pop_config(
        cfg, "save_filename", must_exist=False, default_value="ep{epoch}-ba{batch}-rank{rank}.pt"
    )
    save_interval: Union[str, int] = pop_config(
        cfg, "save_interval", must_exist=False, default_value="1000ba"
    )
    save_num_checkpoints_to_keep: int = pop_config(
        cfg, "save_num_checkpoints_to_keep", must_exist=False, default_value=-1
    )
    progress_bar = pop_config(cfg, "progress_bar", must_exist=False, default_value=False)
    log_to_console: bool = pop_config(cfg, "log_to_console", must_exist=False, default_value=True)
    python_log_level: Optional[str] = pop_config(
        cfg, "python_log_level", must_exist=False, default_value="debug"
    )
    console_log_interval: Union[int, str] = pop_config(
        cfg, "console_log_interval", must_exist=False, default_value="1ba"
    )
    device_train_microbatch_size: Union[str, int] = pop_config(
        cfg, "device_train_microbatch_size", must_exist=False, default_value="auto"
    )
    eval_subset_num_batches: int = pop_config(
        cfg, "eval_subset_num_batches", must_exist=False, default_value=-1
    )
    eval_first: bool = pop_config(cfg, "eval_first", must_exist=False, default_value=False)
    load_path: str = pop_config(cfg, "load_path", must_exist=False, default_value=None)
    load_weights_only: bool = pop_config(
        cfg, "load_weights_only", must_exist=False, default_value=False
    )
    load_strict_model_weights: bool = pop_config(
        cfg, "load_strict_model_weights", must_exist=False, default_value=True
    )
    load_ignore_keys: Optional[List[str]] = pop_config(
        cfg, "load_ignore_keys", must_exist=False, default_value=None
    )
    save_ignore_keys: Optional[List[str]] = pop_config(
        cfg, "save_ignore_keys", must_exist=False, default_value=None
    )
    compile_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "compile_config", must_exist=False, default_value=None
    )
    metadata: Optional[Dict[str, str]] = pop_config(
        cfg, "metadata", must_exist=False, default_value=None, convert=True
    )
    should_log_config: bool = pop_config(cfg, "log_config", must_exist=False, default_value=True)

    num_cpu_threads: Optional[int] = cfg.pop("num_cpu_threads", 0)
    if num_cpu_threads > 0:
        print(f"Setting number of CPU threads to {num_cpu_threads}")
        torch.set_num_threads(num_cpu_threads)
        spops.set_num_threads(num_cpu_threads)

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if (
        logged_cfg.get("run_name", None) is not None
        and save_folder is not None
        and not save_overwrite
        and not save_weights_only
    ):
        autoresume_default = True

    if cfg.get("autoresume") is None and autoresume_default:
        log.info(
            "As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True..."
        )

    autoresume: bool = pop_config(
        cfg, "autoresume", must_exist=False, default_value=autoresume_default
    )

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    pop_config(cfg, "data_local", must_exist=False)
    pop_config(cfg, "data_remote", must_exist=False)
    pop_config(cfg, "global_seed", must_exist=False)
    pop_config(cfg, "global_train_batch_size", must_exist=False)
    pop_config(cfg, "n_gpus", must_exist=False)
    pop_config(cfg, "device_train_grad_accum", must_exist=False)

    assert fsdp_config is None or ds_config is None, "fsdp and deepspeed are not supported together"

    # Warn users for unused parameters
    for key in cfg:
        warnings.warn(
            f"Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary."
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn("FSDP is not applicable for single-GPU training. Reverting to DDP.")
        fsdp_config = None

    # set logging level
    if python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=f"%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s"
        )
        logging.getLogger("llmfoundry").setLevel(python_log_level.upper())  # Foundry module
        logging.getLogger(__name__).setLevel(python_log_level.upper())  # Train script

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({"fsdp_config": fsdp_config}, merge=True)

    # Build tokenizer
    log.info("Building tokenizer...")
    tokenizer_name = tokenizer_config["name"]
    tokenizer_kwargs = tokenizer_config.get("kwargs", {})
    tokenizer_kwargs["num_proc"] = 1
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Scheduler
    scheduler_name: str = scheduler_config.pop("name")
    scheduler = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = (
        [build_logger(str(name), logger_cfg) for name, logger_cfg in logger_configs.items()]
        if logger_configs
        else []
    )

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            # mosaicml_logger will be None if run isn't on MosaicML platform
            loggers.append(mosaicml_logger)

    if metadata is not None:
        # Flatten the metadata for logging
        logged_cfg.pop("metadata", None)
        logged_cfg.update(metadata, merge=True)
        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(metadata)
            mosaicml_logger._flush_metadata(force_flush=True)

    # Profiling
    profiler: Optional[Profiler] = None
    profiler_cfg: Optional[DictConfig] = pop_config(
        cfg, "profiler", must_exist=False, convert=False, default_value=None
    )
    if profiler_cfg:
        profiler_schedule_cfg: Dict = pop_config(
            profiler_cfg, "schedule", must_exist=True, convert=True
        )
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: List[TraceHandler] = []
        profiler_trace_cfg: Optional[Dict] = pop_config(
            profiler_cfg, "json_trace_handler", must_exist=False, default_value=None, convert=True
        )
        if profiler_trace_cfg:
            profiler_trace_handlers.append(JSONTraceHandler(**profiler_trace_cfg))
        profiler = Profiler(
            **profiler_cfg, trace_handlers=profiler_trace_handlers, schedule=profiler_schedule
        )

    # Callbacks
    callbacks: List[Callback] = (
        [
            build_callback(str(name), callback_cfg, om.to_container(logged_cfg))
            for name, callback_cfg in callback_configs.items()
        ]
        if callback_configs
        else []
    )

    use_async_eval = any(isinstance(c, AsyncEval) for c in callbacks)

    print("ROSA CONFIG", rosa_config)
    # Build Model
    print("Initializing model...")
    with init_context:
        assert (
            fsdp_config is None or rosa_config is None
        ), "fsdp is cuurently not supported with RoSA"
        model = build_composer_peft_model(
            model_config, rosa_config, tokenizer, is_fsdp=fsdp_config is not None
        )
        if rosa_config is not None:
            assert isinstance(model.model.base_model, RosaModel)

    # Algorithms
    algorithms = (
        [
            build_algorithm(str(name), algorithm_cfg)
            for name, algorithm_cfg in algorithm_configs.items()
        ]
        if algorithm_configs
        else []
    )

    if rosa_config is not None:
        algorithms.append(RosaScheduler(model.model.base_model))

    # Dataloaders
    log.info("Building train loader...")
    try:
        disable_caching()
        train_loader = build_dataloader(
            train_loader_config,
            tokenizer,
            device_train_batch_size,
        )
    except Exception as e:
        if mosaicml_logger is not None:
            mosaicml_logger.log_exception(e)
        raise e

    if mosaicml_logger is not None:
        mosaicml_logger.log_metrics({"data_validated": time.time()})

    ## Evaluation
    if use_async_eval:
        evaluators = []
        if eval_first:
            warnings.warn("AsyncEval callback does not support eval_first=True. Ignoring.")
            eval_first = False

    else:
        log.info("Building eval loader...")
        eval_icl_seq_len: int = icl_seq_len if icl_seq_len else max_seq_len
        evaluators, _, eval_gauntlet_callback = build_evaluators(
            eval_loader_config,
            icl_tasks_config,
            eval_gauntlet_config,
            tokenizer=tokenizer,
            device_eval_batch_size=device_eval_batch_size,
            icl_seq_len=eval_icl_seq_len,
            icl_subset_num_batches=icl_subset_num_batches,
        )
        if eval_gauntlet_callback is not None:
            callbacks.append(eval_gauntlet_callback)

    if mosaicml_logger is not None:
        log_train_analytics(
            mosaicml_logger,
            model_config,
            train_loader_config,
            eval_loader_config,
            callback_configs,
            tokenizer_name,
            load_path,
            icl_tasks_config,
            eval_gauntlet_config,
        )
    # Log number of parameters
    if hasattr(model, "n_total_params"):
        n_params = model.n_total_params
        n_trainable_params = n_params  # We currently assume all parameters are trainable.
    else:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, "n_active_params"):
        n_active_params = model.n_active_params
    else:
        n_active_params = n_params
    logged_cfg.update(
        {
            "n_params": n_params,
            "n_active_params": n_active_params,
            "n_trainable_params": n_trainable_params,
        }
    )

    # Optimizer
    optimizer_name: str = optimizer_config.pop("name")
    if rosa_config is None or "lora_lr" not in rosa_config:
        optimizer = build_optimizer(model, optimizer_name, optimizer_config)
    else:
        print(f'Using a different learning rate for lora params {rosa_config["lora_lr"]}')
        assert optimizer_name == "decoupled_adamw"
        lora_params = []
        other_params = []
        for name, param in model.named_parameters():
            if any(
                [k in name for k in ["rosa_A", "rosa_B", "rosa_embedding_A", "rosa_embedding_B"]]
            ):
                lora_params.append(param)
            else:
                other_params.append(param)

        print(f"Found {len(lora_params)} lora params and {len(other_params)} other params")
        params = [{"params": other_params}, {"params": lora_params, "lr": rosa_config["lora_lr"]}]
        optimizer = DecoupledAdamW(params, **optimizer_config)

    # Now add the eval metrics
    try:
        if eval_loader_config is not None and not use_async_eval:
            eval_metrics = model.get_metrics(is_train=False)
            non_icl_metrics = [
                metric_name
                for metric_name, metric in eval_metrics.items()
                if not isinstance(metric, InContextLearningMetric)
            ]
            evaluators = add_metrics_to_eval_loaders(evaluators, non_icl_metrics)
    except Exception as e:
        if mosaicml_logger is not None:
            mosaicml_logger.log_exception(e)
        raise e

    # Build the Trainer
    log.info("Building trainer...")
    dtypes = {x.dtype for x in model.parameters()}
    print(dtypes)
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=max_duration,
        eval_interval=eval_interval,
        eval_subset_num_batches=eval_subset_num_batches,
        progress_bar=progress_bar,
        log_to_console=log_to_console,
        console_log_interval=console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=precision,
        algorithms=algorithms,
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=fsdp_config,
        deepspeed_config=ds_config,
        save_folder=save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        load_strict_model_weights=load_strict_model_weights,
        load_ignore_keys=load_ignore_keys,
        save_ignore_keys=save_ignore_keys,
        autoresume=autoresume,
        python_log_level=python_log_level,
        dist_timeout=dist_timeout,
        profiler=profiler,
        compile_config=compile_config,
    )

    if should_log_config:
        log.info("Logging config")
        log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    # Eval first if requested
    if eval_first and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    # Do manual overwriting of the masks directory if they already exist.
    if rosa_config is not None:
        if rosa_config["mask_save_path"] and rosa_config["masks_only"]:
            if os.path.exists(rosa_config["mask_save_path"]):
                print("Overwriting Masks")
                shutil.rmtree(rosa_config["mask_save_path"])

    log.info("Starting training...")
    trainer.fit()

    # Hacky solution for moving the model checkpoint from the
    # subdirectory that the HF writer wrote it into, and into
    # our desired and expected location. Only needed for full
    # (not low-rank) finetuning.
    if rosa_config is None and torch.distributed.get_rank() == 0:
        path_to_save = os.path.join(hf_save_path, run_name)
        hf_output_path = os.path.join(path_to_save, "huggingface")
        for filename in glob.glob(os.path.join(hf_output_path, "*", "*")):
            shutil.copy(filename, path_to_save)
        shutil.rmtree(os.path.join(hf_output_path))

    # if rosa is enabled, save the model manually, since
    # llm-foundry's checkpointing doesn't work properly with RoSA
    if rosa_config is not None:
        assert fsdp_config is None, "fsdp is currently not supported with RoSA"
        path_to_save = os.path.join(hf_save_path, run_name)
        print(f"saving the model to {path_to_save}")
        if torch.distributed.get_rank() == 0:
            model.model.save_pretrained(
                path_to_save, is_main_process=True, state_dict=model.model.state_dict()
            )
            tokenizer.save_pretrained(path_to_save)

    if save_merged_model:
        path_to_save = os.path.join(hf_save_path, run_name, "merged")
        model = model.model.merge_and_unload()
        model.save_pretrained(path_to_save)
        tokenizer.save_pretrained(path_to_save)

    log.info("Done.")
    return trainer


if __name__ == "__main__":
    main()
