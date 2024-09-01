import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from panza3 import PanzaWriter  # The import also loads custom Hydra resolvers
from panza3.entities import EmailInstruction

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


@hydra.main(version_base="1.1", config_path="../configs", config_name="panza_writer")
def main(cfg: DictConfig) -> None:
    LOGGER.info("Starting Panza Writer")
    LOGGER.info("Configuration: \n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Rename config keys to follow class structure
    rename_config_keys(cfg)

    # Instantiate Panza writer
    writer: PanzaWriter = hydra.utils.instantiate(cfg.writer)
    assert isinstance(writer, PanzaWriter), "Failed to instantiate PanzaWriter"

    # TODO: Connect to CLI / GUI / webserver, etc.
    output, prompt = writer.run(
        instruction=EmailInstruction(instruction="Write an email."), return_prompt=True
    )
    print("Prompt:", prompt)
    print("Output:", output)


if __name__ == "__main__":
    main()
