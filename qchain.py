from file_monitor import start_file_monitor
from vector_service import start_vector_monitor
from fastapi import FastAPI, Request, File, UploadFile, WebSocket, Form
from fastapi.responses import JSONResponse, FileResponse
import shutil
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
from kbase_service import create_chat_completion, ChatCompletionRequest, ChatCompletionResponse
from vector_service import recreate_vector_db
import os
from pathlib import Path
from tools.pdf_trans import translate_pdf_html
from starlette.responses import FileResponse
import threading
import asyncio
import nest_asyncio
nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

websockets = {}


@app.get("/")
async def index():
    return {"message": "local knowledge base"}


def translate_pdf_html_bythread(pdf, html, llm, clientid, fn):
    translate_pdf_html(pdf, html, llm, clientid, fn)


@app.post("/upload")
async def upload_file(clientid: str = Form(...), file: UploadFile = UploadFile(...)):
    upload_path = "./uploads"
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    try:
        fileName = f'{upload_path}/{file.filename}'
        with open(fileName, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        sendWebcketMsgSync(clientid, "上传完成")
        t = threading.Thread(target=translate_pdf_html, args=(
            fileName, clientid + ".html", False, clientid, sendWebcketMsgSync,))
        t.start()
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})


async def websocket_handler(websocket: WebSocket, clientid: str):
    await websocket.accept()
    websockets[clientid] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message from client {clientid}: {data}")
    except:
        print(f"WebSocket connection closed for client {clientid}")
    finally:
        del websockets[clientid]


def sendWebcketMsgSync(clientid, msg):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(sendWebcketMsg(clientid, msg))


async def sendWebcketMsg(clientid, msg):
    for clientid, websocket in websockets.items():
        await websocket.send_text(msg)


@app.websocket("/ws/{clientid}")
async def upload_progress(websocket: WebSocket, clientid: str):
    await websocket_handler(websocket, clientid)


@app.get("/download/{file_name}")
async def download(file_name: str):
    f = os.path.join('./', file_name)
    fr = FileResponse(
        path=f,
        filename=Path(f).name,
    )
    return fr


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
