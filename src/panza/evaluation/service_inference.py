import os
import sys
from typing import Annotated

import torch

from fastapi import FastAPI, HTTPException, Header
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting, rag

class Request(BaseModel):
    text: str

class Response(BaseModel):
    generated_text: str

sys.path.pop(0)

app = FastAPI()

origins = [
    "https://mail.google.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from the .env file
load_dotenv()
valid_api_keys = os.getenv("API_KEYS").split(",")

parser = base_inference.get_base_inference_args_parser()
args = parser.parse_args()

print("Running inference with args:", args)

if args.nthreads is not None:
    torch.set_num_threads(args.nthreads)

print("Loading model ", args.model)
model, tokenizer = base_inference.load_model_and_tokenizer(args.model, args.device, args.dtype, load_in_4bit=args.load_in_4bit)

if args.use_rag:
    embeddings_model = rag.get_embeddings_model(args.embedding_model)
    db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

system_preamble, user_preamble, rag_preamble = prompting.load_all_preambles(
    args.system_preamble, args.user_preamble, args.rag_preamble
)

def predict(user_input):
    prompts, outputs = base_inference.run_inference(
        instructions=[user_input],
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

    print("Processed input:", prompts[0])
    print("Generated email", outputs[0])

    return outputs[0]

@app.options('/generate')
def options():
    return {"methods": ["POST"]}

@app.post('/generate')
def generate_text(request: Request, x_api_key: Annotated[str | None, Header()] = None):
    if x_api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key, must be one of: " + str(valid_api_keys))
    generated_text = predict(request.text)
    return {"generated_text": generated_text}
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)