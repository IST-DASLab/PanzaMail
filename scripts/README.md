# Scripts for training and running Panza.

This directory contains all scripts necessary to train and run Panza. We provide brief descriptions, as well as an [advanced user guide to data and hyperparam tuning](#advanced-panza-user-guide).

## Script Descriptions

#### Setup
* `config.sh` sets the necessary environment variables and other parameters used throughout the Panza workflow. This script should be edited by the user in several places: to set the user's email address (for data preprocessing), to select the LLM used for data summarization and Panza finetuning, and optionally to update the locations the data and models will be stored.

#### Data preparation
* `prepare_data.py` does several things:

1. Extracts the user's emails from the `.mbox` file and removes any unusable ones (such as email forwards, those that seem to be written in a foreign language, or those that are too short).
1. Automatically converts emails to training and test data by using an LLM to write their summaries in the form of prompts.
1. Optionally, splits the summarized  into train and test data. This is not done by default because we expect most users to use the default hyperparameters, and therefore have no need for evaluation. To activate this feature, indicate the size of the test split as follows: `python ./prepare_data.py test_split=0.2`
1. Prepares the RAG database. Note that only train data is used for this step.

#### Training
* `train_rosa.sh` performs [parameter-efficient training](https://arxiv.org/pdf/2401.04679.pdf). 
* `train_fft.sh` performs full-parameter/full-rank training. _Note that this requires additional computational resources (about 2x)._ 


#### Inference/Serving

Serving is done through the `runner` object. To use the runner, the type of model and the type of interface must be specified.

For interfaces, we offer serving via CLI (command-line inference) and an online GUI (via Gradio), as well as a bulk-serving API via JSON for the JSON, the location of the file defaults to the test data, but can be overridden (see the "evaluation" section, below).

Currently, we support full-finetuned and parameter-efficienty-finetuned models. These must be set through the `writer-llm` parameter. 
* To serve a foundation (i.e., not locally-finetuned) model or a fully-finetuned model, set `writer/llm=transformers`
* To serve a PEFT model, set `writer/llm=peft`

Thus, a serving command would look something like:

```
python runner.py user=[username] interfaces=[cli|gui] writer/llm=[peft|transformers] checkpoint=[checkpoint_loc]
```

For the json interface, it would look like:

```
python runner.py user=[username] interfaces=json writer/llm=[peft|transformers] checkpoint=[checkpoint_loc] interfaces.input_file=[json_file_loc]
```

##### Evaluation

We think of evaluation as a special form of bulk inference/serving. Thus, like other forms of inference, it is done through a runner, specifically through the `json` interface.

A sample command that runs interface over the test set looks like:

```
python runner.py user=jen interfaces=json writer/llm=[peft|transformers] checkpoint=[checkpoint_loc] interfaces.input_file=../data/test.jsonl
```


# Complete guide on running inference, including on CPU

There are two backend options: Ollama (no GPU) or Local (with GPU). The dependencies necessary for each backend are different.

### Inference using Transformers

#### Setup

If you are running inference on the same machine/setup that you trained on, no additional setup is required. Otherwise, from the root directory of the Panza project, run:

```bash
pip install -e .
```
and 
```bash
pip install panza_mail[training]

```

#### Inference

Inference can be run as follows.

- To run Panza after a full training run, try something like `CUDA_VISIBLE_DEVICES=0 ./runner.sh user=USERNAME interfaces=cli writer/llm=transformers`.
- To run Panza after a RoSA or LoRA training run, replace `writer/llm=transformers` with `writer/llm=peft`.


### CPU inference with Ollama

#### Setup

In the environment in which you plan to do inference, from the root directory of the Panza project, run:

```bash
pip install -e .

```

#### Model preparation


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

#### Run Panza

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


## Advanced Panza User Guide

### Data Guide

:bulb: We recommend having between 128 and 1000 sent emails as training targets. Less than 128 might cause the model to overfit, while we haven't found that more than 1000 emails help for the style transfer. However, we encourage you to include as many emails as available in the RAG database, as they will provide the model with additional context. To sub-select training data, you can perform the usual flow with all of your data (export, run `extract_emails.sh` and `prepare_dataset.sh`), and then simply remove all but your target number of rows from the resulting `train.jsonl` in the `data`.

:bulb: To merge data from multiple mailboxes (such as combining your personal and work emails), run `extract_emails.sh` on each `.mbox` file, remembering to change the value of `user.email_address` and `user.user_name` in `config.sh` for every inbox. Then simply concatenate the resulting `[user.user_name].clean.jsonl` files to one, and use that file's `user.user_name` going forward. Make sure that the `prepare_dataset.sh` script is run _after_ the merge with `force_extract_clean_emails=false`.


### Hyper-Parameter Tuning Guide

To get the most out of Panza, it is essential to find good hyper-parameters for the fine-tuning process. 
Specifically the key parameters to consider are the learning rates (`trainer.optimizer.lr=0.1` and  `trainer.optimizer.rosa.lora_lr`, in the case of RoSA fine-tuning) and (`trainer.optimizer.max_duration`) parameters, whose values should be adjusted based on your amount of data and model in use. 

Here are some general guidelines for hyper-parameter fine-tuning: 

* In our experience, a good target for the Perplexity over the training set (displayed during and at the end of the training run) is in the range 1-1.5 (for full fine-tuning) to 2-3 (for RoSA tuning). At that point, Panza should be able to reproduce your writing style quite faithfully.
* To reach this target, you can ajust two parameters: the length of training (`trainer.optimizer.max_duration`) and the learning rates (`trainer.optimizer.lr` for full fine-tuning and `trainer.optimizer.lr` and `trainer.optimizer.rosa.lora_lr` for RoSA).
* Specifically, for full fine-tuning we have found 3 training epochs to be sufficient. For RoSA fine-tuning, one usually needs 5-7 epochs for best results. 
* Regarding the learning rates, we have already provided stable default values (around 1e-5 for LLaMA3-8B , Phi-3.5-mini, and Mistral). You may adjust these depending on the amount of your local data.
* We have found that setting these values too low will yield default "impersonal'' answers (specifically, the same answers as the base model with some context). Setting them too high will lead the model to "overfit" to the user data, to the point where a lot of the latent model "knowledge" is lost. The key to good performance is to find a good middle ground between these two scenarios.  
