import logging

import glob
import hydra
import os
from omegaconf import DictConfig, OmegaConf

from panza import PanzaWriter  # The import also loads custom Hydra resolvers

LOGGER = logging.getLogger(__name__)


def rename_config_keys(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    cfg.writer.llm.sampling_parameters = cfg.writer.llm.sampling
    del cfg.writer.llm.sampling

    cfg.writer.prompt_builder = cfg.writer.prompting
    del cfg.writer.prompting

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)


def set_latest_model(cfg: DictConfig) -> None:
    model_files = glob.glob(
        f"{cfg.checkpoint_dir}/models/*"
    )  # * means all if need specific format then *.csv
    latest_file = max(model_files, key=os.path.getctime)

    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint = latest_file
    OmegaConf.set_struct(cfg, True)


@hydra.main(version_base="1.1", config_path="../configs", config_name="panza_writer")
def main(cfg: DictConfig) -> None:
    LOGGER.info("Starting Panza Writer")
    # Add value from interfaces into writer config. We want the interface to choose whether the prompt is to be returned or not.
    if "remove_prompt_from_stream" in cfg.interfaces:
        OmegaConf.set_struct(cfg, False)
        cfg.writer.llm.remove_prompt_from_stream = cfg.interfaces.remove_prompt_from_stream
        del cfg.interfaces.remove_prompt_from_stream
        OmegaConf.set_struct(cfg, True)
    LOGGER.info("Configuration: \n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Rename config keys to follow class structure
    rename_config_keys(cfg)
    # Find the latest checkpoint, if requested.
    if cfg.checkpoint == "latest":
        set_latest_model(cfg)

    # Instantiate Panza writer
    writer: PanzaWriter = hydra.utils.instantiate(cfg.writer)
    assert isinstance(writer, PanzaWriter), "Failed to instantiate PanzaWriter"

    # Instantiate interfaces (CLI, GUI, web, etc) as specified in the configuration
    hydra.utils.instantiate(cfg.interfaces, writer=writer)


if __name__ == "__main__":
    main()
