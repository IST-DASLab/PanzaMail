set -e

current_user=$(whoami)

composer ../src/panza3/finetuning/train.py \
    user=${current_user} finetuning=full