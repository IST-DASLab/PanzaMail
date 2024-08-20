import os
import sys
from typing import Annotated

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting
from panza.evaluation import ollama_inference

class Request(BaseModel):
    text: str

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

system_preamble, user_preamble, rag_preamble, _ = prompting.load_all_preambles(
    args.system_preamble, args.user_preamble, args.rag_preamble, args.thread_preamble
)

def predict(user_input):
    relevant_emails = []
    prompt = prompting.create_prompt(
            user_input,
            system_preamble,
            user_preamble,
            rag_preamble,
            relevant_emails,
        )
    return ollama_inference.get_response_stream(prompt, args.model)

def streamer(stream):
    for chunk in stream:
        yield chunk["message"]["content"]

@app.options('/generate')
def options():
    return {"methods": ["POST"]}

@app.post('/generate')
def generate_text(request: Request, x_api_key: Annotated[str | None, Header()] = None):
    if x_api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    stream = predict(request.text)
    return StreamingResponse(streamer(stream), media_type='text/event-stream')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5001)