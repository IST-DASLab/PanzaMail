import argparse
import os
import sys

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting, rag

sys.path.pop(0)


def get_execute(model, tokenizer, system_preamble, user_preamble, rag_preamble, db, args):

    def execute(prompt):
        prompt, output = base_inference.run_inference(
            instruction=prompt,
            model=model,
            tokenizer=tokenizer,
            system_preamble=system_preamble,
            user_preamble=user_preamble,
            rag_preamble=rag_preamble,
            rag_relevance_threshold=args.rag_relevance_threshold,
            rag_num_emails=args.rag_num_emails,
            use_rag=args.use_rag,
            db=db if args.use_rag else None,
            max_new_tokens=args.max_new_tokens,
            best=args.best,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
        cleaned = output.split("[/INST]")[-1].strip()
        cleaned = cleaned.split("</s>")[0]
        print("Prompt\n", prompt)
        print("Output\n", cleaned)
        yield cleaned

    return execute


def main():
    parser = base_inference.get_base_inference_args_parser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    print("Running inference with args:", args)

    if args.nthreads is not None:
        torch.set_num_threads(args.nthreads)

    print("Loading model ", args.model)
    model, tokenizer = base_inference.load_model_and_tokenizer(args.model, args.device, args.dtype)

    if args.use_rag:
        embeddings_model = rag.get_embeddings_model(args.embedding_model)
        db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    system_preamble, user_preamble, rag_preamble = prompting.load_all_preambles(
        args.system_preamble, args.user_preamble, args.rag_preamble
    )

    with gr.Blocks() as panza:
        gr.Markdown("# Panza\n")
        inputbox = gr.Textbox(label="Input", placeholder="Enter text and press ENTER")
        outputbox = gr.Textbox(label="Output", placeholder="Generated result from the model")
        inputbox.submit(
            get_execute(
                model=model,
                tokenizer=tokenizer,
                system_preamble=system_preamble,
                user_preamble=user_preamble,
                rag_preamble=rag_preamble,
                db=db if args.use_rag else None,
                args=args,
            ),
            [inputbox],
            [outputbox],
        )

    panza.queue().launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()
