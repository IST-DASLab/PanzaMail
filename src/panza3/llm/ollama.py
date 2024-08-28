import os
import ollama

from .base import LLM

class OllamaLLM(LLM):
    def __init__(self, name: str, gguf_file: str, sampling_params: dict):
        """
        Loads and serves the model from the GGUF file into Ollama with the given name and sampling parameters.
        """
        super().__init__(name)
        self.gguf_file = gguf_file
        self.sampling_params = sampling_params

        if not self._is_ollama_running():
            self._start_ollama()

        if not self._is_model_loaded():
            self._load_model()

    def _is_ollama_running(self):
        try:
            ollama.list()
            return True
        except:
            return False
    
    def _start_ollama(self):
        # run the bash command "ollama list" which causes Ollama to start if it is not already running
        try:
            os.system("/bin/bash -c 'ollama list'")
        except:
            raise Exception("Ollama failed to start.")

    def _is_model_loaded(self):
        for model in ollama.list()['models']:
            # model name is everything before the colon
            name = model['name'].split(":")[0]
            if name == self.name:
                return True
        return False
    
    def _load_model(self):
        # TODO: Add sampling parameters to the model file
        modelfile = f"""
        FROM {self.gguf_file}
        """
        try:
            ollama.create(model={self.name}, modelfile=modelfile, stream=True)
        except:
            raise Exception(f"Failed to load model {self.name} with GGUF file {self.gguf_file}.")
        
    def _get_message(self, response):
        return response['message']['content']

    def chat(self, messages):
        response = ollama.chat(model=self.name, messages=messages, stream=False)
        return self._get_message(response)

    def chat_stream(self, messages):
        stream = ollama.chat(
            model=self.name,
            messages=messages,
            stream=True,
        )
        # return a new stream that only contains the message content
        for chunk in stream:
            yield self._get_message(chunk)