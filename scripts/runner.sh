# Convenience script for launching your fine-tuned model.
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# CUDA_VISIBLE_DEVICES=x ./runner.sh user=USERNAME interfaces=cli writer/llm=transformers

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

python3 runner.py ${vars[@]}