# How to run inference in Panza3

There are two backend options: Ollama (no GPU) or Local (with GPU). The dependencies necessary for each backend are different.

## Step 1: Install Dependencies for Panza

```bash
pip install -e .

pip install ollama

```

For Local, run:
```bash
pip install -e .
```
and 
```bash
pip install panza_mail[training]
```

## Step 2a: Ollama Prerequisites

If running with Ollama, then Ollama needs to be installed from the [web page](https://ollama.com/).

On a server, this is done with
```bash
curl -fsSL https://ollama.com/install.sh > install.sh
```

Then, you will need to convert your model into a GGUF file.

to do that, first install (with `pip install -e.`) [llama.cpp](https://github.com/ggerganov/llama.cpp)

Then train a model with the flag `finetuning.save_merged_model=true`

Then, go to the llama.cpp source folder and run a command like:

```bash
python ../../llama.cpp/convert_hf_to_gguf.py /nfs/scistore19/alistgrp/eiofinov/PanzaMail/checkpoints/models/test_gguf_3/merged --outfile /tmp/custom.gguf --outtype q8_0
```

The resulting model file can be copied to your local machine.


## Step 3: Set configurations

In the `configs folder` add a user YAML file for yourself in `/user`.

If running with Ollama, edit the `name` and `gguf` fields in `/writer/llm/ollama.yaml` with a name of your choice and the path to the GGUF file.
Note that as of right now, if the model changes but the name does not, the new model will not be loaded.

## Step 4: Run Panza

To run Panza, cd into the `scripts` directory and run:
```bash
python3 runner.py user=<your name> interfaces=<cli/gui/web> writer/llm=<ollama/peft/transformers>
```
For example, to run with Ollama and the CLI interface with the user `test`, run:
```bash
python3 runner.py user=test interfaces=cli writer/llm=ollama
```