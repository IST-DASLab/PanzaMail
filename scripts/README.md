# Scripts for training and running Panza.

This directory contains all scripts necessary to train and run Panza. We provide brief descriptions, as well as an [advanced user guide to data and hyperparam tuning](#advanced-panza-user-guide).

## Script Descriptions

#### Setup
* `config.sh` sets the necessary environment variables and other parameters used throughout the Panza workflow. This script should be edited by the user in several places: to set the user's email address (for data preprocessing), to select the LLM used for data summarization and Panza finetuning, and optionally to update the locations the data and models will be stored.

#### Data preparation
* `extract_emails.sh` extracts the user's emails from the `.mbox` file and removes any unusable ones (such as email forwards, those that seem to be written in a foreign language, or those that are too short).
* `prepare_dataset.sh` automatically converts emails to training data by using an LLM to write their summaries in the form of prompts; it then splits them into train and test data, and prepares the RAG database.

#### Training
* `train_rosa.sh` performs [parameter-efficient training](https://arxiv.org/pdf/2401.04679.pdf), and evaluation. For evaluation, we use a heldout email dataset and compute the BLEU score between the output email and the one originally written by the user. 
* `train_fft.sh` performs full-parameter/full-rank training, and then evaluation (as before). _Note that this requires additional computational resources (about 2x)._ 

#### Serving
* `run_panza_cli.sh` runs a simple tool in the command line that enables a user to put in prompts and get Panza responses.
* `run_panza_gui.sh` runs a simple tool in the browser that enables a user to put in prompts and get Panza responses.

Both of these tools require a link to the model that you wish to use. Running without providing a `MODEL` argument will run inference on the base (non-finetuned) LLM.

```
./run_panza_gui.sh MODEL=<path-to-your-trained-model>
```


## Advanced Panza User Guide

### Data Guide

:bulb: We recommend having between 128 and 1000 sent emails as training targets. Less than 128 might cause the model to overfit, while we haven't found that more than 1000 emails help for the style transfer. However, we encourage you to include as many emails as available in the RAG database, as they will provide the model with additional context. To sub-select training data, you can perform the usual flow with all of your data (export, run `extract_emails.sh` and `prepare_dataset.sh`), and then simply remove all but your target number of rows from the resulting `train.jsonl` in the `data`.

:bulb: To merge data from multiple mailboxes (such as combining your personal and work emails), run `extract_emails.sh` on each `.mbox` file, remembering to change the value of `PANZA_EMAIL_ADDRESS` in `config.sh` for every inbox. Then simply concatenate the resulting `[email_id].clean.jsonl` files to one, and use that file's `email_id` for the `PANZA_EMAIL_ADDRESS` argument in `config.sh` going forward. Make sure that the `prepare_dataset.sh` script is run _after_ the merge.


### Hyper-Parameter Tuning Guide

To get the most out of Panza, it is essential to find good hyper-parameters for the fine-tuning process. 
Specifically the key parameters to consider are the learning rates (`LR` and  `LORA_LR`, in the case of RoSA fine-tuning) and (`NUM_EPOCHS`) parameters, whose values should be adjusted based on your amount of data and model in use. 

Here are some general guidelines for hyper-parameter fine-tuning: 

* In our experience, a good target for the Perplexity over the training set (displayed during and at the end of the training run) is in the range 1-1.5 (for full fine-tuning) to 2-3 (for RoSA tuning). At that point, Panza should be able to reproduce your writing style quite faithfully.
* To reach this target, you can ajust two parameters: the length of training (`NUM_EPOCHS`) and the learning rates (`LR` for full fine-tuning and `LR` and `LORA_LR` for RoSA).
* Specifically, for full fine-tuning we have found 3 training epochs to be sufficient. For RoSA fine-tuning, one usually needs 5-7 epochs for best results. 
* Regarding the learning rates, we have already provided stable default values (around 1e-5 for both LLaMA3-8B and Mistral). You may adjust these depending on the amount of your local data.
* We have found that setting these values too low will yield default "impersonal'' answers (specifically, the same answers as the base model with some context). Setting them too high will lead the model to "overfit" to the user data, to the point where a lot of the latent model "knowledge" is lost. The key to good performance is to find a good middle ground between these two scenarios.  
