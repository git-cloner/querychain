from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import threading
import schedule
import time
from file_monitor import file_monitor_queue
import sys
import os
from tqdm import tqdm
from config import base_dir, index_dir, embedding_model
# import shutil

indexing_flag = False
vectordb_search = None


def load_docs_from_dir(directory):
    loader = DirectoryLoader(directory, glob='**/*.*', show_progress=True)
    documents = loader.load()
    return documents


def load_docs_from_file(filename):
    loader = UnstructuredFileLoader(filename)
    documents = loader.load()
    return documents


def split_docs(documents):
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


def create_vectorstore(split_docs):
    embeddings = SentenceTransformerEmbeddings(
        model_name=embedding_model)
    vectordb = Chroma.from_documents(
        split_docs, embeddings, persist_directory=index_dir)
    return vectordb


def load_vectorstore():
    global vectordb_search
    embeddings = SentenceTransformerEmbeddings(
        model_name=embedding_model)
    vectordb_search = Chroma(
        embedding_function=embeddings, persist_directory=index_dir)


def search_docs(query):
    matching_docs = vectordb_search.similarity_search(query)
    return matching_docs


def remove_folder(path):
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            for filename in os.listdir(path):
                remove_folder(os.path.join(path, filename))
            os.rmdir(path)


def recreate_vector_db():
    # remove vdb
    remove_folder(index_dir)
    # recreate vdb
    directory = base_dir
    documents = load_docs_from_dir(directory)
    splitdocs = split_docs(documents)
    if len(splitdocs) > 0:
        vectordb = create_vectorstore(splitdocs)
        vectordb.persist()
        vectordb_search = vectordb
    return


def adddoc_to_vector_db(filename):
    if not os.path.exists(filename):
        return
    documents = load_docs_from_file(filename)
    if len(documents) > 0:
        splitdocs = split_docs(documents)
        vectordb = create_vectorstore(splitdocs)
        vectordb.persist()
        vectordb_search = vectordb
    return


def job_file():
    global indexing_flag
    if indexing_flag:
        return
    len = file_monitor_queue.qsize()
    if len == 0:
        return
    indexing_flag = True
    progress = tqdm(total=len)
    while file_monitor_queue.qsize() > 0:
        _filename = file_monitor_queue.get()
        adddoc_to_vector_db(_filename)
        progress.update(1)
    indexing_flag = False
    progress.close()


def start_watchdog():
    print("向量库监控开启：" + index_dir)
    schedule.every(1).minutes.do(job_file)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(1)


def start_vector_monitor():
    load_vectorstore()
    worker = threading.Thread(target=start_watchdog)
    worker.start()
