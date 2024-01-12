from file_monitor import start_file_monitor
from vector_service import start_vector_monitor
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
from kbase_service import create_chat_completion, ChatCompletionRequest, ChatCompletionResponse
from vector_service import recreate_vector_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return {"message": "local knowledge base"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    return create_chat_completion(request)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reindex', action='store_true',
                        default=False, required=False)
    args = parser.parse_args()
    if args.reindex:
        recreate_vector_db()
    start_file_monitor()
    start_vector_monitor()
    uvicorn.run(app, host='0.0.0.0', port=8060, workers=1)
