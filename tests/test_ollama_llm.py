from typing import Type
from unittest.mock import patch, MagicMock

from panza3.llm.ollama import OllamaLLM
import pytest

MODEL = "test_model"
GGUF_FILE = "test.gguf"
SAMPLING_PARAMS = {"param1": "val1"}
REQUEST = "write an email"
RESPONSE = "here is an email"
RESPONSE_OBJ = {
  'message': {
    'content': RESPONSE
  }
}

@patch('os.system')
@patch('ollama.list')
def test_ollama_llm_init_launches_ollama(ollama_list: MagicMock, os_system: MagicMock):
  # When Ollama isn't running, the __init__() should start it by calling os.system(). To simulate Ollama not running, we'll mock the ollama.list() method to raise an exception.
  ollama_list.side_effect = Exception("Ollama not running")
  try:
    OllamaLLM("test", "test.gguf", {})
  except:
    pass
  os_system.assert_called_once_with("/bin/bash -c 'ollama list'")

@patch('ollama.create')
@patch('ollama.list')
def test_ollama_llm_init_creates_model(ollama_list: MagicMock, ollama_create: MagicMock):
  # When the given module isn't loaded into Ollama yet, the __init__() should load it by calling ollama.create(). To simulate the module not being loaded, we'll mock the ollama.list() method to return an empty list.
  ollama_list.return_value = {'models': []}
  OllamaLLM(MODEL, GGUF_FILE, SAMPLING_PARAMS)
  ollama_create.assert_called_once()

# Mock all external calls to prevent side effects
@patch('os.system')
@patch('ollama.list')
@patch('ollama.create')
def test_ollama_llm_init(*args):
  # Make sure __init__() sets all local variables correctly
  ollama_llm = OllamaLLM(MODEL, GGUF_FILE, SAMPLING_PARAMS)
  assert ollama_llm.gguf_file == GGUF_FILE
  assert ollama_llm.sampling_params == SAMPLING_PARAMS
  assert ollama_llm.name == MODEL

# Mock all external calls to prevent side effects
@patch('os.system')
@patch('ollama.list')
@patch('ollama.create')
@patch('ollama.chat')
def test_ollama_llm_chat(ollama_chat: MagicMock, *args):
  ollama_chat.return_value = RESPONSE_OBJ
  ollama_llm = OllamaLLM(MODEL, GGUF_FILE, SAMPLING_PARAMS)
  assert ollama_llm.chat(REQUEST) == [RESPONSE]
  ollama_chat.assert_called_once()

# Mock all external calls to prevent side effects
@patch('os.system')
@patch('ollama.list')
@patch('ollama.create')
@patch('ollama.chat')
def test_ollama_llm_chat_stream(ollama_chat: MagicMock, *args):
  expected_iterator = iter([RESPONSE_OBJ])
  ollama_chat.return_value = expected_iterator
  ollama_llm = OllamaLLM(MODEL, GGUF_FILE, SAMPLING_PARAMS)
  # make sure that ollama_llm.chat() returns a generator that yields the expected response
  assert list(ollama_llm.chat_stream(REQUEST)) == [RESPONSE]
  ollama_chat.assert_called_once_with(model=MODEL, messages=REQUEST, stream=True)