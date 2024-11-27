from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Type

import torch

_MISSING_LIBRARIES = []

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    AutoPeftModelForCausalLM = None
    _MISSING_LIBRARIES.append("peft")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _MISSING_LIBRARIES.append("transformers")

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
    _MISSING_LIBRARIES.append("bitsandbytes")


from .base import LLM, ChatHistoryType


class LocalLLM(LLM):
    def __init__(
        self,
        name: str,
        checkpoint: str,
        device: str,
        sampling_parameters: Dict,
        dtype: str,
        load_in_4bit: bool,
        remove_prompt_from_stream: bool,
    ):
        self._check_installation()

        super().__init__(name, sampling_parameters)
        self.checkpoint = checkpoint
        self.device = device

        assert dtype in [None, "fp32", "bf16"]
        if device == "cpu":
            assert dtype == "fp32", "CPU only supports fp32, please specify --dtype fp32"
        dtype = None if dtype is None else (torch.float32 if dtype == "fp32" else torch.bfloat16)
        self.dtype = dtype

        self.remove_prompt_from_stream = remove_prompt_from_stream
        self.load_in_4bit = load_in_4bit
        self.quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if load_in_4bit
            else None
        )

        self._load_model_and_tokenizer()

    def chat(self, messages: ChatHistoryType | List[ChatHistoryType]) -> List[str]:
        encodeds = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_dict=True,
        )
        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **self.sampling_parameters,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prompt_length = encodeds["input_ids"].shape[1]
        outputs = self.tokenizer.batch_decode(
            generated_ids[:, prompt_length:], skip_special_tokens=True
        )

        return outputs

    def chat_stream(self, messages: ChatHistoryType) -> Iterator[str]:
        if isinstance(messages[0], (list, tuple)) or hasattr(messages[0], "messages"):
            raise TypeError("chat_stream does not support batched messages.")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=self.remove_prompt_from_stream,
            skip_special_tokens=self.remove_prompt_from_stream,
        )
        encodeds = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_dict=True,
        )
        model_inputs = encodeds.to(self.device)
        generation_kwargs = dict(
            **model_inputs,
            **self.sampling_parameters,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        from threading import Thread

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def _check_installation(self) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is not installed. Please install it with `pip install transformers`."
            )

        if BitsAndBytesConfig is None:
            from transformers import __version__ as version

            raise ImportError(
                f"transformers {version} does not support 4-bit quantization. Please upgrade to a newer version."
            )

    def _load_model_and_tokenizer_with_constructor(self, model_class: Type[Any]) -> None:
        if self.load_in_4bit:
            self.model = model_class.from_pretrained(
                self.checkpoint,
                device_map=self.device,
                quantization_config=self.quantization_config,
                trust_remote_code=True,
            )
        else:
            self.model = model_class.from_pretrained(
                self.checkpoint,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, model_max_length=self.model.config.max_position_embeddings
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def _load_model_and_tokenizer(self) -> None:
        pass


class TransformersLLM(LocalLLM):
    def _load_model_and_tokenizer(self):
        self._load_model_and_tokenizer_with_constructor(AutoModelForCausalLM)


class PeftLLM(LocalLLM):
    def _check_installation(self) -> None:
        super()._check_installation()
        if AutoPeftModelForCausalLM is None:
            raise ImportError("peft is not installed.")

    def _load_model_and_tokenizer(self) -> None:
        self._load_model_and_tokenizer_with_constructor(AutoPeftModelForCausalLM)
        self.model = self.model.merge_and_unload()
