# How to run inference in Panza3

There are two backend options: Ollama (no GPU) or Local (with GPU). The dependencies necessary for each backend are different.

## Inference using Transformers

### Setup

If you are running inference on the same machine/setup that you trained on, no additional setup is required. Otherwise, from the root directory of the Panza project, run:

```bash
pip install -e .
```
and 
```bash
pip install panza_mail[training]

```

### Inference

Inference can be run as follows.

- To run Panza after a full training run, try something like `CUDA_VISIBLE_DEVICES=0 ./runner.sh user=USERNAME interfaces=cli writer/llm=transformers`.
- To run Panza after a RoSA or LoRA training run, replace `writer/llm=transformers` with `writer/llm=peft` TODO Armand: can we fix this? 


## Inference with Ollama (CPU)

### Setup

In the environment in which you plan to do inference, from the root directory of the Panza project, run:

```bash
pip install -e .

```

### Model preparation


If running with Ollama, then Ollama needs to be installed from the [web page](https://ollama.com/).

Then, you will need to convert your model into a GGUF file.

to do that, first install [llama.cpp](https://github.com/ggerganov/llama.cpp) following the instructions in the README.

 An existing model can be converted with the [Data Preparation Guide](./merge_and_convert_to_gguf.sh). Note that for a model that uses parameter-efficient finetuning (LoRA or RoSA), a new model file is created with the adapter weights merged into the bases model weights.

Then, go to the llama.cpp source folder and run a command like:

The resulting `.gguf` model file can be copied to the machine where you plan to run inference.


### Set configuration

Ensure that, on the machine performing inference, the user profile and the prompt preambles are set up similarly to the training machine. If doing inference with RAG, you will also need to copy over all of the associated files in the `data/` folder.

To configure the Ollama writer, edit the `name` and `gguf` fields in `/writer/llm/ollama.yaml` with a name of your choice and the path to the GGUF file.

Note that as of right now, if the model changes but the name does not, the new model will not be loaded.

### Run Panza

To run Panza, from the `scripts/` directory, run:
```bash
python3 runner.py user=<your name> interfaces=<cli/gui/web> writer/llm=ollama
```

For example, to run with Ollama and the CLI interface with the user `test`, run:
```bash
python3 runner.py user=test interfaces=cli writer/llm=ollama
```

### :hammer_and_wrench: Troubleshooting Ollama
In some setups, we have seen errors when trying to create the Ollama model with `ollama.create` called during the writer command above. If you encounter this issue, please follow these steps to create the model directly with Ollama from the CLI, and then rerun the writer script.

1. Create a Modelfile - In the `ollama.py` file, we specify a model file as a String that looks something like this:
    ```
    FROM [insert path to model here]
    PARAMETER temperature 0.7
    PARAMETER top_k 50
    PARAMETER top_p 0.7
    PARAMETER num_predict 1024
    ```
    The sampling parameters (where we specify the temperature) are optional.
2. Create the model with Ollama - To do so, write execute the following command in the terminal `ollama create [name_of_model] -f [path_to_modelfile]`. If successful, you will be able to see that the model has been successfully created with `ollama list`.
3. Run the Panza runner as before! Since the model has now been created, the Panza script will be able to pick this up and use the created model for inference.