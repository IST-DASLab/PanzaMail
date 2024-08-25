from typing import Dict, Iterator, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    ):
        super().__init__(name, sampling_parameters)
        self.checkpoint = checkpoint
        self.device = device

        assert dtype in [None, "fp32", "bf16"]
        if device == "cpu":
            assert dtype == "fp32", "CPU only supports fp32, please specify --dtype fp32"
        dtype = None if dtype is None else (torch.float32 if dtype == "fp32" else torch.bfloat16)
        self.dtype = dtype

        self.load_in_4bit = load_in_4bit
        # TODO: Add conditional import for BitsAndBytesConfig?
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

    def _load_model_and_tokenizer(self) -> None:
        pass


class TransformersLLM(LocalLLM):

    def _load_model_and_tokenizer(self):
        if self.load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                device_map=self.device,
                quantization_config=self.quantization_config,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
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

        # TODO: Implement chat_stream.
        raise NotImplementedError("chat_stream is not implemented for TransformersLLM.")
