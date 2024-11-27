# Script to convert a model to the GGUF format with 8-bit quantization.
# If the model was trained with the PEFT module (e.g., using RoSA),
# the model is first merged, which also leaves a merged model as an artifact.
# The GGUF model is written to the same folder as the original model; or
# to [original_folder]/merged if the adapter is merged as a part of running
# this script.

# Usage: ./merge_and_convert_to_gguf.sh path/to/model/or/adapter


GGUF_TYPE=q8_0
GGUF_MODEL_NAME=custom

# Check if model needs to be merged
if [ -e $1/adapter_config.json ]; then
   python merge_adapters.py $1
   model_folder=$1/merged
else
   model_folder=$1
fi

python ../../llama.cpp/convert_hf_to_gguf.py ${model_folder} --outfile ${model_folder}/${GGUF_MODEL_NAME}.gguf --outtype ${GGUF_TYPE}
