import codecs
import logging
import os
import pty
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import hydra
import psutil
import torch
from omegaconf import DictConfig, OmegaConf

from panza3 import PanzaWriter  # The import also loads custom Hydra resolvers

LOGGER = logging.getLogger(__name__)


def create_run_name(cfg: DictConfig) -> str:
    # export RUN_NAME=panza_${PANZA_USERNAME}_${MODEL_TYPE}_${MODEL_PRECISION}-bs${BS}-fft-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}${PREAMBLE_STR}${RAFT_STR}-$RANDOM

    run_name = f"panza_{cfg.user.username}"

    model_name = cfg.model.split("/")[-1]
    run_name += f"-{model_name}"

    run_name += f"-{cfg.model_precision}"
    run_name += f"-bs{cfg.batch_size}"

    if hasattr(cfg.finetuning, "rosa"):
        run_name += "-rosa"
    else:
        run_name += "-fft"

    run_name += f"-lr{cfg.lr}"
    run_name += f"-epochs{cfg.num_epochs}"
    run_name += f"-seed{cfg.seed}"
    run_name += f"-{random.randint(1e6, 1e7 - 1)}"

    return run_name


def determine_rosa_schedule(cfg: DictConfig) -> str:
    pass


def create_experiment_yaml() -> str:
    pass


def create_checkpoint_dirs(cfg: DictConfig) -> None:
    # Create model directory
    os.makedirs(cfg.finetuning.hf_save_path)

    # Create mask directory
    if hasattr(cfg.finetuning, "rosa"):
        os.makedirs(cfg.finetuning.rosa.mask_save_path)


def get_hf_save_precision(cfg: DictConfig) -> str:
    if cfg.model_precision == "bf16":
        return "bfloat16"
    elif cfg.model_precision == "fp32":
        return "fp32"
    else:
        raise ValueError(f"Unsupported model_precision: {cfg.model_precision}")


def get_rosa_dtype(cfg: DictConfig) -> str:
    if cfg.model_precision == "bf16":
        return "bg16"
    elif cfg.model_precision == "fp32":
        return "fp32"
    elif cfg.model_precision == "4bit":
        return "fp32"
    else:
        raise ValueError(f"Unsupported model_precision: {cfg.model_precision}")


def override_config(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    cfg.finetuning.run_name = create_run_name(cfg)

    if hasattr(cfg.finetuning, "rosa"):
        pass
    else:
        cfg.finetuning.callbacks.hf_checkpointer.precision = get_hf_save_precision(cfg)

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)


def save_config_to_yaml(cfg: DictConfig) -> str:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as temp_file:
        OmegaConf.save(config=cfg, f=temp_file.name)
        return temp_file.name


def launch_experiment(cfg: DictConfig, finetuning_yaml: str, prompt_builder_yaml: str) -> None:
    def terminate_process_tree(pid: str):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                print("Terminating child process", child)
                child.terminate()
            psutil.wait_procs(children, timeout=5)
            print("Terminating parent process", parent)
            parent.terminate()
            parent.wait(5)
        except psutil.NoSuchProcess:
            pass

    def move_checkpoint_files(cfg: DictConfig) -> None:
        # Move checkpoint files to the final directory
        run_save_path = Path(cfg.hf_save_path) / "models" / cfg.run_name
        huggingface_dir = run_save_path / "huggingface"
        last_save_dir_name = max(huggingface_dir.iterdir(), key=os.path.getmtime).name

        # Move the contents of the last saved directory to the run save path
        source_dir = huggingface_dir / last_save_dir_name
        for item in source_dir.iterdir():
            shutil.move(str(item), run_save_path)

        # Remove the now-empty huggingface directory
        shutil.rmtree(huggingface_dir)

    train_script = os.path.join(cfg.panza_workspace, "src/panza3/finetuning/train.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.path.join(cfg.panza_workspace, "src")
    environment["WANDB_PROJECT"] = f"panza-{cfg.user.username}"
    environment["WANDB_DISABLED"] = str(int(cfg.wandb_disabled))
    environment["PANZA_PREPROCESSING_CONFIG"] = prompt_builder_yaml

    print(finetuning_yaml)
    print(train_script)
    print(environment["PYTHONPATH"])
    command = f"composer {train_script} {finetuning_yaml}"
    master, slave = pty.openpty()  # Open a pseudo-terminal
    with subprocess.Popen(
        command,
        stdout=slave,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        preexec_fn=os.setsid,
        shell=True,
    ) as process:
        os.close(slave)  # Close the slave descriptor

        # Set up a stream reader for the master end of the pty
        try:
            with codecs.getreader("utf-8")(os.fdopen(master, "rb")) as reader:
                # Read and process output line by line
                for line in reader:
                    print(line, end="")

            return process.returncode
        except KeyboardInterrupt:
            print("Killing process")
            # os.killpg(os.getpgid(process.pid), subprocess.signal.SIGTERM)
            terminate_process_tree(process.pid)
            torch.cuda.empty_cache()
            time.sleep(3)  # Give some time for GPU resources to be released

        if not hasattr(cfg.finetuning, "rosa"):
            move_checkpoint_files(cfg)

        print("Find the finetuned model at", os.path.join(cfg.hf_save_path, "models", cfg.run_name))


@hydra.main(version_base="1.1", config_path="../configs", config_name="panza_finetuning")
def main(cfg: DictConfig) -> None:
    LOGGER.info("Starting Panza Finetuning")
    LOGGER.info("Configuration: \n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Override configuration
    override_config(cfg)

    # Launch training
    if "rosa" in cfg.finetuning:
        pass
    else:
        finetuning_yaml = save_config_to_yaml(cfg.finetuning)
        preprocessing_yaml = save_config_to_yaml(cfg.preprocessing)
        launch_experiment(cfg, finetuning_yaml, preprocessing_yaml)


if __name__ == "__main__":
    main()
