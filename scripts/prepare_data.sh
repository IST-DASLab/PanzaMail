# Convenience script for data preparation
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# CUDA_VISIBLE_DEVICES=x ./prepare_data.sh user=alonso

set -e

vars=()
idx=1

# process input arguments
for argument in "$@"
do
   key=$(echo $argument | cut -f1 -d=)
    vars[idx]=$argument
    idx+=1
done

python ./prepare_data.py ${vars[@]}