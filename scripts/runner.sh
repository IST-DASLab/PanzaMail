# Convenience script for launching your fine-tuned model.
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# CUDA_VISIBLE_DEVICES=x ./runner.sh user=USERNAME interfaces=cli writer/llm=transformers

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
   else
    vars[idx]=$argument
    idx+=1
   fi
done

python3 runner.py ${vars[@]}