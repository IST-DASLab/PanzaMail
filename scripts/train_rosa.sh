# Convenience script for running RoSA finetuning.
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# ./train_rosa.sh user=alonso trainer.optimizer.lr=0.1

set -e

vars=()
idx=1

# process input arguments
for argument in "$@"
do
   key=$(echo $argument | cut -f1 -d=)
   
   if [[ $key == finetuning ]]; then
    echo "The 'finetuning' argument is already set and should not be overridden here; override is ignored."
   elif [[ $key == finetuning.rosa.masks_only ]]; then
    echo "The 'finetuning.rosa.masks_only' argument is already set and should not be overridden here; override is ignored."
   else
    vars[idx]=$argument
    idx+=1
   fi
done

# First create the masks for RoSA finetuning.
composer ../src/panza/finetuning/train.py \
    finetuning=rosa finetuning.rosa.masks_only=true ${vars[@]}

# Then train the weights.
composer ../src/panza/finetuning/train.py \
    finetuning=rosa finetuning.rosa.masks_only=false ${vars[@]}
