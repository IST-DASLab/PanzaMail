<div align="center">
  <img src="panza_logo.png" alt="panza logo" width="300"/>
</div>

# Panza: A personal email assistant, trained and running on-device

##  What is Panza?
Panza is an assistant tool for automatically generating emails that are customized to your writing style and past email history. \
Panza takes a set of emails you have sent and a publicly-available Large Language Model (LLM), and produces a fine-tuned LLM that matches your writing style, pairing it with a Retrieval-Augmented Generation (RAG) component which helps it produce relevant emails.
Panza **can be trained and run entirely locally, on a single GPU with
less than 24 GiB of memory**.
Training and execution are also quick - for a dataset on the order of 1000 emails, training Panza takes well under an hour, and generating a new email takes a few seconds at most.

One key part of Panza is a dataset-generation technique we call **data playback**:  Given some already sent emails, we create a training set for Panza by using a pretrained LLM to summarize the emails in instruction form; each email becomes an (instruction, email) pair.
Given a dataset consisting of all pairs, we use these pairs to "play back" your sent emails: the LLM receives only the instruction, and has to generate the "ground truth" email as a training target.
We find that this approach is very useful for the LLM to ``learn'' the user's writing style. Further, we pair it with a RAG system over the email data, which helps retrieve common information, such as links or addresses.

## How it works

### :film_projector: Step 1: Data playback

For most email clients, it is possible to download a user's past emails in a machine-friendly .mbox format. For example, GMail allows you to do this via [Google Takeout](takeout.google.com).
We translate sent emails locally into training data for Panza, as follows. For each email, Panza uses a pretrained LLM to write an instruction that should lead a customized personal assistant LLM to generate the email. This gives us a (semi-synthetic) training set of *(instruction; real email)* pairs that we use to finetune a LLM to generate emails that match those written by the user, in response to the synthetic prompts created in the previous step. We call this technique *data playback*.


### :weight_lifting: Step 2: Local Fine-tuning via Robust Adaptation (RoSA)

We then use parameter-efficient finetuning to train the LLM on this dataset, locally. We found that we get the best results with the [RoSA method](https://arxiv.org/pdf/2401.04679.pdf), which combines low-rank (LoRA) and sparse finetuning. If parameter efficiency is not a concern, that is, you have a more powerful GPU, then regular, full-rank/full-parameter finetuning can also be used. We find that a moderate amount of further training strikes the right balance between matching the writer's style without memorizing irrelevant details in past emails.


### :owl:	Step 3: Serving via RAG

Once we have a custom user model, Panza can be run locally together with a Retrieval-Augmented Generation (RAG) module. Specifically, this functionality stores past emails in a database and provides a few relevant emails as context for each new query. This allows Panza to better insert specific details, such as a writer's contact information or frequently used Zoom links.

<div align="center">
  <img src="panza_diagram.png" alt="panza logo" width="703" style="max-width: 100%; height: auto;"/>
</div>

## Installation

### Conda
1. Make sure you have a version of [conda](https://docs.anaconda.com/free/miniconda/miniconda-install/) installed.
2. Run `source prepare_env.sh`. This script will create a conda environment named `panza` and install the required packages.

### Docker
Run the following commands to pull a docker image with all the dependencies installed.
```
docker pull istdaslab/panzamail
docker run -it --gpus all istdaslab/panzamail /bin/bash
```

or alternatively, you can build the image yourself:
```
docker build . -f Dockerfile -t istdaslab/panzamail
docker run -it --gpus all istdaslab/panzamail /bin/bash
```

In the docker you can activate the `panza` environment with:
```
micromamba activate panza
```

## :rocket: Getting started

To quickly get started with building your own personalized email assistant, follow the steps bellow:

<!-- To train your personalized email assistant, follow the three steps below. -->

<!-- TODO: Replace steps with #### heading? -->
### Step 0: Download your sent emails
<!-- **Step 1: Download your sent emails** -->
<details>
  <summary> Expand for detailed download instructions.</summary>

  We provide a description for doing this for GMail via Google Takeout.

  1. Go to [https://takeout.google.com/](https://takeout.google.com/).
  2. Click `Deselect all`.
  3. Find `Mail` section (search for the phrase `Messages and attachments in your Gmail account in MBOX format`).
  4. Select it.
  5. Click on `All Mail data included` and deselect everything except `Sent`.
  6. Scroll to the bottom of the page and click `Next step`.
  7. Click on `Create export`.
  8. Wait for download link to arrive in your inbox.
  9. Download `Sent.mbox` and place it in the `data/` directory.

  For Outlook accounts, we suggest doing this via a Thunderbird plugin for exporting a subset of your email as an MBOX format, such as [this add-on](https://addons.thunderbird.net/en-us/thunderbird/addon/importexporttools-ng/).
</details>

At the end of this step you should have the downloaded emails placed inside `data/Sent.mbox`.

<!-- **Step 0: Environment configuration** -->
### Step 1: Environment configuration

<!-- 🎛️ -->
Panza is configured through a set of environment variables defined in `scripts/config.sh` and shared along all running scripts.

<!-- 💬 -->
The LLM prompt is controlled by a set of `prompt_preambles` that give the model more insight about its role, the user and how to reuse existing emails for *Retrieval-Augmented Generation (RAG)*.

:warning: Before continuing, make sure you complete the following setup:
  - Modifiy the environment variable `PANZA_EMAIL_ADDRESS` inside `scripts/config.sh` with your own email address.
  - Modifiy `prompt_preambles/user_preamble.txt` with your own information. If you choose, this can even be empty.
  - Login to Hugging Face to be able to download pretrained models: `huggingface-cli login` (no charge).
  - Login to Weights & Biases to log metrics during training: `wandb login` (no charge).

You are now ready to move to `scripts`.
``` bash
cd scripts
```

### Step 2: Extract emails
<!-- **Step 2: Extract emails** -->

1. Run `./extract_emails.sh`. This extracts your emails in text format to `data/<username>_clean.jsonl` which you can manually inspect.

2. If you wish to eliminate any emails from the training set (e.g. containing certain personal information), you can simply remove the corresponding rows.

### Step 3: Prepare dataset
<!-- **Step 3: Prepare dataset** -->

1. Simply run `./prepare_dataset.sh`.<details>
    <summary> This scripts takes care of all the prerequisites before training (expand for details). </summary>

    - Creates synthetic prompts for your emails as described in the [data playback](#film_projector-step-1-data-playback) section. The results are stored in `data/<username>_clean_summarized.jsonl` and you can inspect the `"summary"` field.
    - Splits data into training and test subsets. See `data/train.jsonl` and `data/test.jsonl`.
    - Creates a vector database from the embeddings of the training emails which will later be used for *Retrieval-Augmented Generation (RAG)*. See `data/<username>.pkl` and `data/<username>.faiss`.
    </details>

### Step 4: Train a LLM on your emails
<!-- **Step 4: Train a LLM on your emails** -->

1. [Recommended] For parmeter efficient fine-tuning, run `./train_rosa.sh`.  
If a larger GPU is available and full-parameter fine-tuning is possible, run `./train_fft.sh`.

Experiment with different hyper-parameters by passing extra arguments to the training script, such as `LR`, `LORA_LR`, `NUM_EPOCHS`. All the trained models are saved in the `checkpoints` directory.

Example:
``` bash
./train_rosa.sh LR=1e-6 LORA_LR=1e-6 NUM_EPOCHS=7
```

### Step 5: Launch Panza!
<!-- **Step 5: Launch Panza!** -->

1. Run `./run_panza.sh MODEL=<path-to-your-trained-model>` to serve the trained model in a friendly GUI.  
Alternatively, if you prefer using the CLI to interact with Panza, run `./run_interactive_cli_inference.sh` instead.

You can experiment with the following arguments:
- If `MODEL` is not specified, it will use a pretrained `Mistral-7B-Instruct-v0.2` model by default. Try it out to compare the syle difference!
- To disable RAG, run with `PANZA_DISABLE_RAG_INFERENCE=1`.

Example:
``` bash
./run_panza.sh \
  MODEL=/home/armand/repos/PanzaMail/checkpoints/models/panza-rosa_1e-6-seed42_7908 \
  PANZA_DISABLE_RAG_INFERENCE=0  # this is the default behaviour, so you can omit it
```

:email: **Have fun with your new email writing assistant!** :email:

<!-- For in depth customization of each step of the pipeline, refer to ... -->
