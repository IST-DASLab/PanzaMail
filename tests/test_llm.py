import pytest

from panza3.llm import TransformersLLM


def test_transformers_llm_init(generative_model: str):
    model = TransformersLLM(
        name="huggingface_model",
        checkpoint=generative_model,
        device="cpu",
        sampling_parameters={"do_sample": False, "max_new_tokens": 50},
        dtype="fp32",
        load_in_4bit=False,
    )
    assert model is not None
    assert model.name == "huggingface_model"
    assert model.checkpoint == generative_model
    assert model.model is not None
    assert model.tokenizer is not None


def test_transformers_llm_generate(generative_model: str):
    model = TransformersLLM(
        name="huggingface_model",
        checkpoint=generative_model,
        device="cpu",
        sampling_parameters={"do_sample": False, "max_new_tokens": 50},
        dtype="fp32",
        load_in_4bit=False,
    )

    messages = [{"role": "user", "content": "Write something."}]

    outputs = model.chat(messages)

    assert outputs is not None
    assert len(outputs) == 1


def test_transformers_llm_generate_batch(generative_model: str):
    model = TransformersLLM(
        name="huggingface_model",
        checkpoint=generative_model,
        device="cpu",
        sampling_parameters={"do_sample": False, "max_new_tokens": 50},
        dtype="fp32",
        load_in_4bit=False,
    )

    messages = [
        [{"role": "user", "content": "Write something."}],
        [{"role": "user", "content": "Write something else."}],
        [{"role": "user", "content": "Write something different."}],
    ]

    outputs = model.chat(messages)

    assert outputs is not None
    assert len(outputs) == 3
