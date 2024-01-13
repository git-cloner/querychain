from langchain_community.llms import OpenAI
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
from config import base_dir, index_dir, embedding_model,splitter
from common_splitter import CommonDocLoader
from policy_splitter import PolicyDocLoader

indexing_flag = False
vectordb_search = None


def get_all_files(directory):
    files = []
    for filepath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(filepath, filename))
    return files


def load_and_split(filename):
    loader = None
    if splitter == "POLICY" :
        loader = PolicyDocLoader(filename)
    else :
        loader = CommonDocLoader(filename)
    return loader.load_and_split()


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
    files = get_all_files(directory)
    progress = tqdm(total=len(files))
    for _filename in files :
        _tmp = os.path.basename(_filename)
        progress.set_description(f"{_tmp}")
        adddoc_to_vector_db(_filename)
        progress.update(1)
    return


def adddoc_to_vector_db(filename):
    if not os.path.exists(filename):
        return
    if filename.startswith("~") :
        return
    try:
        split_docs = load_and_split(filename)
        if len(split_docs) > 0:
            vectordb = create_vectorstore(split_docs)
            vectordb.persist()
            vectordb_search = vectordb
    except Exception as e:
        print(e)
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
