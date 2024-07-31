MODEL_PATH=
OUT_PATH=
GGUF_TYPE=q8_0

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # if not set, default to 0

python merge_adapters_bf16.py --model_path ${MODEL_PATH}
python convert_hf_to_gguf.py ${MODEL_PATH}/merged/ --outfile ${OUT_PATH} --outtype ${GGUF_TYPE}
