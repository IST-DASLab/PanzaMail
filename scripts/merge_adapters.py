import os.path
import torch
import fire
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main(model_path):
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    out_path = os.path.join(model_path, "merged")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print("Model saved to", out_path)


if __name__ == "__main__":
    fire.Fire(main)
