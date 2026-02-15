import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    get_scheduler,
    set_seed,
)

from panza import PanzaWriter  # The import also loads custom Hydra resolvers.


def create_lora_run_name(cfg: DictConfig) -> str:
    run_name = f"panza_{cfg.user.username}"
    model_name = cfg.finetuning.model_name_or_path.split("/")[-1]
    run_name += f"-{model_name}"
    run_name += f"-{cfg.model_precision}"
    run_name += f"-bs{cfg.finetuning.batch_size}"
    run_name += "-lora"
    run_name += f"-lr{cfg.finetuning.lr}"
    run_name += f"-{cfg.finetuning.max_duration}"
    run_name += f"-seed{cfg.finetuning.seed}"
    return run_name


def parse_num_epochs(max_duration: Any) -> float:
    if isinstance(max_duration, (int, float)):
        return float(max_duration)
    if isinstance(max_duration, str) and max_duration.endswith("ep"):
        return float(max_duration[:-2])
    raise ValueError(
        f"Unsupported finetuning.max_duration value: {max_duration}. "
        "For LoRA HF training, use values like '5ep'."
    )


def parse_warmup_steps(t_warmup: Any) -> int:
    if isinstance(t_warmup, int):
        return t_warmup
    if isinstance(t_warmup, str) and t_warmup.endswith("ba"):
        return int(t_warmup[:-2])
    return 0


def save_config_to_yaml(cfg: DictConfig) -> str:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as temp_file:
        OmegaConf.save(config=cfg_dict, f=temp_file.name)
        return temp_file.name


def get_model_dtype(cfg: DictConfig) -> torch.dtype:
    if cfg.model_precision == "bf16":
        return torch.bfloat16
    if cfg.model_precision == "fp32":
        return torch.float32
    if cfg.model_precision == "4bit":
        return torch.bfloat16
    raise ValueError(f"Unsupported model_precision: {cfg.model_precision}")


def get_quantization_config(model_cfg: DictConfig) -> Optional[BitsAndBytesConfig]:
    weight_bias_dtype = model_cfg.get("weight_bias_dtype", None)
    if weight_bias_dtype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def tokenize_example(example: Dict[str, Any], tokenizer: Any, max_seq_len: int) -> Dict[str, Any]:
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class CausalLMDataCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels: List[List[int]] = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_masks.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig,
    base_lr: float,
    lora_lr: Optional[float],
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_cfg.get("name", "decoupled_adamw")
    if optimizer_name != "decoupled_adamw":
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}' for HF LoRA training. "
            "Use decoupled_adamw."
        )

    betas_cfg = optimizer_cfg.get("betas", [0.9, 0.999])
    betas: Tuple[float, float] = (float(betas_cfg[0]), float(betas_cfg[1]))
    eps = float(optimizer_cfg.get("eps", 1e-8))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if lora_lr is None:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]):
            lora_params.append(param)
        else:
            other_params.append(param)

    if not lora_params:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    param_groups: List[Dict[str, Any]] = [{"params": lora_params, "lr": float(lora_lr)}]
    if other_params:
        param_groups.insert(0, {"params": other_params, "lr": base_lr})

    return torch.optim.AdamW(
        param_groups,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


@hydra.main(version_base="1.1", config_path="../../../configs", config_name="panza_finetuning")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    if "lora" not in cfg.finetuning:
        raise ValueError("This trainer only supports finetuning=lora.")
    if "rosa" in cfg.finetuning:
        raise ValueError("This trainer does not support RoSA. Use scripts/train_rosa.sh instead.")

    if not cfg.finetuning.run_name:
        cfg.finetuning.run_name = create_lora_run_name(cfg)
    OmegaConf.resolve(cfg)

    cfg.preprocessing.model = cfg.finetuning.model_name_or_path
    if "retriever" in cfg.preprocessing.prompting:
        # LoRA training does not require RAG retrieval and should not require FAISS assets.
        cfg.preprocessing.prompting.retriever = OmegaConf.create(
            {"_target_": "panza.retriever.NoneRetriever"}
        )
    preprocessing_yaml = save_config_to_yaml(cfg.preprocessing)

    os.environ["PANZA_PREPROCESSING_CONFIG"] = preprocessing_yaml
    os.environ["WANDB_PROJECT"] = f"panza-{cfg.user.username}"
    os.environ["WANDB_DISABLED"] = str(int(cfg.finetuning.wandb_disabled))

    set_seed(int(cfg.finetuning.seed))

    from panza.finetuning.preprocessing import panza_preprocessing_function

    train_file = os.path.join(cfg.user.data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at {train_file}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.finetuning.model_name_or_path,
        model_max_length=cfg.finetuning.max_seq_len,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = get_model_dtype(cfg)
    quantization_config = get_quantization_config(cfg.finetuning.model)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.finetuning.model_name_or_path,
        torch_dtype=model_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        use_cache=False,
        device_map="auto" if quantization_config is not None else None,
        attn_implementation="eager",
    )

    lora_cfg = cfg.finetuning.lora
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=train_file, split="train")
    train_dataset = train_dataset.map(
        panza_preprocessing_function,
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )
    train_dataset = train_dataset.map(
        lambda example: tokenize_example(
            example,
            tokenizer=tokenizer,
            max_seq_len=int(cfg.finetuning.max_seq_len),
        ),
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )

    per_device_train_batch_size = int(cfg.finetuning.get("device_train_microbatch_size", 1))
    target_batch_size = int(cfg.finetuning.batch_size)
    gradient_accumulation_steps = max(1, math.ceil(target_batch_size / per_device_train_batch_size))

    num_train_epochs = parse_num_epochs(cfg.finetuning.max_duration)
    warmup_steps = parse_warmup_steps(cfg.finetuning.scheduler.get("t_warmup", 0))

    output_dir = os.path.join(cfg.finetuning.hf_save_path, cfg.finetuning.run_name)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(cfg.finetuning.lr),
        num_train_epochs=num_train_epochs,
        bf16=(cfg.finetuning.precision == "amp_bf16"),
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
        report_to=[] if cfg.finetuning.wandb_disabled else ["wandb"],
        remove_unused_columns=False,
    )

    optimizer = build_optimizer(
        model=model,
        optimizer_cfg=cfg.finetuning.optimizer,
        base_lr=float(cfg.finetuning.lr),
        lora_lr=(
            float(cfg.finetuning.lora.lora_lr)
            if "lora_lr" in cfg.finetuning.lora
            else None
        ),
    )

    num_update_steps_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / (per_device_train_batch_size * gradient_accumulation_steps)),
    )
    num_training_steps = max(1, math.ceil(num_train_epochs * num_update_steps_per_epoch))
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id),
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if bool(cfg.finetuning.get("save_merged_model", False)):
        merged_dir = os.path.join(output_dir, "merged")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    os.remove(preprocessing_yaml)


if __name__ == "__main__":
    main()
