import os
from typing import Dict, Iterator, List
from .base import LLM, ChatHistoryType

_MISSING_LIBRARIES = []

try:
    import ollama
except ImportError:
    ollama = None
    _MISSING_LIBRARIES.append("ollama")


class OllamaLLM(LLM):
    def __init__(
        self,
        name: str,
        gguf_file: str,
        sampling_parameters: Dict,
        overwrite_model: bool,
        remove_prompt_from_stream: bool,
    ):
        """
        Loads and serves the model from the GGUF file into Ollama with the given name and sampling parameters.
        """
        super().__init__(name, sampling_parameters)
        self.gguf_file = gguf_file
        self.sampling_parameters = sampling_parameters
        self.overwrite_model = overwrite_model
        self.remove_prompt_from_stream = remove_prompt_from_stream  # Ollama removes prompt for us so this is stored to match with signature.

        if not self._is_ollama_running():
            self._start_ollama()

        if not self._is_model_loaded():
            self._load_model()

    def _is_ollama_running(self) -> bool:
        try:
            ollama.list()
            return True
        except:
            return False

    def _start_ollama(self) -> None:
        # run the bash command "ollama list" which causes Ollama to start if it is not already running
        try:
            os.system("/bin/bash -c 'ollama list'")
        except:
            raise Exception("Ollama failed to start.")

    def _is_model_loaded(self) -> bool:
        for model in ollama.list()["models"]:
            # model name is everything before the colon
            name = model["model"].split(":")[0]
            if name == self.name:
                if self.overwrite_model:
                    print(
                        f"Model {self.name} already exists, but we are deleting the old copy and creating a new on."
                    )
                    ollama.delete(model=name)
                else:
                    print(f"Model {self.name} already exists; not recreating")
                    return True
        return False

    def _make_modelfile_parameters(self) -> str:
        if self.sampling_parameters is None or self.sampling_parameters["do_sample"] == False:
            return ""
        return "PARAMETER temperature 0.7\nPARAMETER top_k 50\nPARAMETER top_p 0.7\nPARAMETER num_predict 1024"

    def _load_model(self) -> None:
        modelfile = f"FROM {self.gguf_file}\n{self._make_modelfile_parameters()}"
        for resp in ollama.create(model=self.name, modelfile=modelfile, stream=True):
            print(resp)
        print("Loaded a new mode into Ollama", self.name)

    def _get_message(self, response) -> str:
        return response["message"]["content"]

    def _check_installation(self) -> None:
        if ollama is None:
            raise ImportError(
                "The 'ollama' library is not installed. Please install it with 'pip install ollama'."
            )

    def chat(self, messages: ChatHistoryType | List[ChatHistoryType]) -> List[str]:
        response = ollama.chat(model=self.name, messages=messages, stream=False)
        return [self._get_message(response)]

    def chat_stream(self, messages: ChatHistoryType) -> Iterator[str]:
        stream = ollama.chat(
            model=self.name,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            yield self._get_message(chunk)
