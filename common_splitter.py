from typing import List, Optional, Any, Iterable
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.docstore.document import Document


class CommonDocLoader(BaseLoader):
    def __init__(
        self,
        filename: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        self.filename = filename
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        loader = UnstructuredFileLoader(self.filename)
        documents = loader.load()
        return documents

    def load_and_split(self):
        documents = self.load()
        text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        split_docs = text_splitter.split_documents(documents)
        return split_docs
