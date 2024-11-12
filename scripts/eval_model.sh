# Convenience script for running full finetuning.
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# ./train_fft.sh user=alonso trainer.optimizer.lr=0.1

set -e

vars=()
# Set a default for the required user argument. We'll override it
# later if provided.
vars[1]=$"user=$(whoami)"
idx=2

# process input arguments
for argument in "$@"
do
   key=$(echo $argument | cut -f1 -d=)
   
   if [[ $key == user ]]; then
    # We already set the default value here; change it now.
    vars[1]=$argument
   elif [[ $key == finetuning ]]; then
    echo "The 'finetuning' argument is already set and should not be overridden here; override is ignored."
   else
    vars[idx]=$argument
    idx+=1
   fi
done

composer ../src/panza3/finetuning/train.py \
    finetuning=full ${vars[@]}