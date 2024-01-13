from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import LLMResult
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import time
import os
from config import openid_base_url, llm_model_id
from vector_service import search_docs
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = '0000'
os.environ["OPENAI_API_BASE"] = openid_base_url
client = OpenAI(api_key="EMPTY", base_url=openid_base_url)


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    # Additional parameters
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class ChatCompletionResponse(BaseModel):
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class ChatOpenAICustomSyncHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.tokens.append("[DONE]")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.tokens.append("[DONE]")

    def get_tokens(self):
        if len(self.tokens) > 0:
            return self.tokens.pop(0)


def load_llmmodel():
    chatOpenAICustomSyncHandler = ChatOpenAICustomSyncHandler()
    llm = ChatOpenAI(model_name=llm_model_id,
                     streaming=True, callbacks=[chatOpenAICustomSyncHandler])
    return llm, chatOpenAICustomSyncHandler


def answer_from_llm(llm, matching_docs, query):
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


def llm_chat_stream(query):
    messages = [
        {
            "role": "system",
            "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return client.chat.completions.create(
        model=llm_model_id,
        messages=messages,
        stream=True,
        max_tokens=512,
        temperature=0.8,
        presence_penalty=1.1,
        top_p=0.8)


def yield_Context(model_id, new_content):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=new_content),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))


def predict_chunk_head(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=""),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    return chunk


def predict_chunk_content(model_id, new_content):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=new_content),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    return chunk


def predict_chunk_stop(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=""),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    return chunk


def predict(query: str, history: List[List[str]], model_id: str):
    llm, chatOpenAICustomSyncHandler = load_llmmodel()
    # push head chunk
    chunk = predict_chunk_head(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # push content chunk
    matching_docs = search_docs(query)
    if (len(matching_docs) > 0):
        # answer from vector_db top 5
        i = 0
        for doc in matching_docs:
            i = i + 1
            if i > 5:
                break
            new_text = "[" + str(i) + "]" + doc.page_content + \
                "[" + doc.metadata["source"] + "]" + "\n\n"
            chunk = predict_chunk_content(model_id, new_text)
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        # answer prepare from llm
        new_text = "以下为大语言模型的返回结果：\n\n"
        chunk = predict_chunk_content(model_id, new_text)
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        # answer from llm
        answer_from_llm(llm, matching_docs, query)
        while True:
            new_token = chatOpenAICustomSyncHandler.get_tokens()
            if new_token == "[DONE]":
                break
            chunk = predict_chunk_content(model_id, new_token)
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
            time.sleep(0.05)
    else:
        response = llm_chat_stream(query)
        if response:
            for _chunk in response:
                new_token = _chunk.choices[0].delta.content
                chunk = predict_chunk_content(model_id, new_token)
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    # push stop chunk
    chunk = predict_chunk_stop(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    generate = predict(query, history, request.model)
    return EventSourceResponse(generate, media_type="text/event-stream")
