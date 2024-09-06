# How to run inference in Panza3

There are two backend options: Ollama (no GPU) or Local (with GPU). The dependencies necessary for each backend are different.

## Step 1: Install Dependencies for Panza

For Ollama, simply run:
```bash
pip install -e .
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

Then, you will need to convert your model into a GGUF file.

## Step 2b: Local Prerequisites

If running locally, then the Panza model needs to be located in `data`.

## Step 3: Set configurations

In the `configs folder` add a user YAML file for yourself in `/user`.

If running with Ollama, edit the `name` and `gguf` fields in `/writer/llm/ollama.yaml` with a name of your choice and the path to the GGUF file.

## Step 4: Run Panza

To run Panza, cd into the `scripts` directory and run:
```bash
python3 runner.py user=<your name> interfaces=<cli/gui/web> writer/llm=<ollama/peft/transformers>
```
For example, to run with Ollama and the CLI interface with the user `test`, run:
```bash
python3 runner.py user=test interfaces=cli writer/llm=ollama
```