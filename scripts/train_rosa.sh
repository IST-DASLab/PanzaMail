set -e

current_user=$(whoami)

# First create the masks for RoSA finetuning.
composer ../src/panza3/finetuning/train.py \
    user=${current_user} finetuning=rosa finetuning.rosa.masks_only=true

# Then train the weights.
composer ../src/panza3/finetuning/train.py \
    user=${current_user} finetuning=rosa finetuning.rosa.masks_only=false