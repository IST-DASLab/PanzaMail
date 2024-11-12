import argparse
import json
import random
from datetime import datetime
from os import makedirs
from os.path import join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-data-dir", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-type", type=str, default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    makedirs(args.output_data_dir, exist_ok=True)

    with open(args.data_path, "r") as f:
        data = f.readlines()

    if args.split_type == "random":
        random.seed(args.seed)
        random.shuffle(data)
    elif args.split_type == "chronological":
        data = sorted(data, key=lambda x: datetime.fromisoformat(json.loads(x)["date"]))
    else:
        raise ValueError("Invalid split type.")

    train_size = int(len(data) * args.train_ratio)

    with open(join(args.output_data_dir, "train.jsonl"), "w") as f:
        for i in range(train_size):
            f.write(data[i])

    with open(join(args.output_data_dir, "test.jsonl"), "w") as f:
        for i in range(train_size, len(data)):
            f.write(data[i])


if __name__ == "__main__":
    main()
