import os
from typing import Annotated, Generator, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from panza.entities.instruction import EmailInstruction, Instruction
from panza.writer import PanzaWriter
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
import threading


class Request(BaseModel):
    text: str


class PanzaWebService:
    def __init__(self, writer: PanzaWriter, port: int):
        self.app = FastAPI()
        self.writer = writer
        self.port = port
        self._setup_routes()
        load_dotenv()
        self._add_cors()
        self.api_keys = self._get_valid_api_keys()
        self._start_server()
        self.server_thread = None

    def _add_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _get_valid_api_keys(self) -> List[str]:
        return os.getenv("API_KEYS").split(",")

    def _streamer(self, stream):
        for chunk in stream:
            yield chunk

    def _predict(self, input: str) -> Generator:
        instruction: Instruction = EmailInstruction(input)
        stream: Generator = self.writer.run(instruction, stream=True)
        return stream

    def _setup_routes(self):
        @self.app.options("/generate")
        def options():
            return {"methods": ["POST"]}

        @self.app.post("/generate")
        def generate_text(request: Request, x_api_key: Annotated[str | None, Header()]):
            if x_api_key not in self.api_keys:
                raise HTTPException(status_code=401, detail="Invalid API key.")
            stream = self._predict(request.text)
            return StreamingResponse(self._streamer(stream), media_type="text/event-stream")

    def _start_server(self):
        self.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"port": self.port},
            daemon=False,
        )
        self.server_thread.start()
        print("Panza web server started.")

    def _stop_server(self):
        if self.server_thread is None:
            return
        self.server_thread.join()
        self.server_thread = None
        print("Panza web server stopped.")
