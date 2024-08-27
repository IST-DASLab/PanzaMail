from omegaconf import OmegaConf

from .prompting.utils import load_preamble, load_user_preamble

OmegaConf.register_new_resolver("load_preamble", load_preamble)
OmegaConf.register_new_resolver("load_user_preamble", load_user_preamble)

from .writer import PanzaWriter

__all__ = ["PanzaWriter"]
