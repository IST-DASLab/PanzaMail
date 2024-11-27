from typing import Type

import pytest
from torch import float32 as torch_float32

from panza.llm import PeftLLM, TransformersLLM
from panza.llm.local import _MISSING_LIBRARIES, LocalLLM

skip_if_no_transformers = pytest.mark.skipif(
    "transformers" in _MISSING_LIBRARIES, reason="transformers is not installed"
)
skip_if_no_peft = pytest.mark.skipif("peft" in _MISSING_LIBRARIES, reason="peft is not installed")
skip_if_no_bitsandbytes = pytest.mark.skipif(
    "bitsandbytes" in _MISSING_LIBRARIES, reason="bitsandbytes is not installed"
)


@pytest.mark.parametrize(
    "local_llm_class, checkpoint",
    [
        pytest.param(
            TransformersLLM, "microsoft/Phi-3-mini-4k-instruct", marks=skip_if_no_transformers
        ),
        # TODO: Replace local Peft checkpoint with fixture
        pytest.param(
            PeftLLM,
            "/nfs/scistore19/alistgrp/Checkpoints/Panza/shared/armand/models/test_rosa_checkpoint",
            marks=[skip_if_no_transformers, skip_if_no_peft],
        ),
    ],
)
def test_local_llm_init(local_llm_class: Type[LocalLLM], checkpoint: str):
    model = local_llm_class(
        name="local_llm",
        checkpoint=checkpoint,
        device="cpu",
        sampling_parameters={"do_sample": False, "max_new_tokens": 50},
        dtype="fp32",
        load_in_4bit=False,
        remove_prompt_from_stream=False,
    )
    assert model is not None
    assert model.name == "local_llm"
    assert model.checkpoint == checkpoint
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.model.device.type == "cpu"
    assert model.dtype == torch_float32
    assert model.model.dtype == model.dtype


@pytest.mark.parametrize(
    "local_llm_class, checkpoint",
    [
        pytest.param(
            TransformersLLM, "microsoft/Phi-3-mini-4k-instruct", marks=skip_if_no_transformers
        ),
        # TODO: Replace local Peft checkpoint with fixture
        pytest.param(
            PeftLLM,
            "/nfs/scistore19/alistgrp/Checkpoints/Panza/shared/armand/models/test_rosa_checkpoint",
            marks=[skip_if_no_transformers, skip_if_no_peft],
        ),
    ],
)
def test_local_llm_generate(local_llm_class: Type[LocalLLM], checkpoint: str):
    model = local_llm_class(
        name="local_llm",
        checkpoint=checkpoint,
        device="cpu",
        sampling_parameters={"do_sample": False, "max_new_tokens": 50},
        dtype="fp32",
        load_in_4bit=False,
        remove_prompt_from_stream=False,
    )

    messages = [{"role": "user", "content": "Write something."}]

    outputs = model.chat(messages)

    assert outputs is not None
    assert len(outputs) == 1


@pytest.mark.parametrize(
    "local_llm_class, checkpoint",
    [
        pytest.param(
            TransformersLLM, "microsoft/Phi-3-mini-4k-instruct", marks=skip_if_no_transformers
        ),
        # TODO: Replace local Peft checkpoint with fixture
        pytest.param(
            PeftLLM,
            "/nfs/scistore19/alistgrp/Checkpoints/Panza/shared/armand/models/test_rosa_checkpoint",
            marks=[skip_if_no_transformers, skip_if_no_peft],
        ),
    ],
)
def test_local_llm_generate_batch(local_llm_class: Type[LocalLLM], checkpoint: str):
    model = local_llm_class(
        name="local_llm",
        checkpoint=checkpoint,
        device="cpu",
        sampling_parameters={"do_sample": False, "max_new_tokens": 50},
        dtype="fp32",
        load_in_4bit=False,
        remove_prompt_from_stream=False,
    )

    messages = [
        [{"role": "user", "content": "Write something."}],
        [{"role": "user", "content": "Write something else."}],
        [{"role": "user", "content": "Write something different."}],
    ]

    outputs = model.chat(messages)

    assert outputs is not None
    assert len(outputs) == 3
