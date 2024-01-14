from typing import List, Optional, Any, Iterable
import argparse
import re
import copy
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


class CommonchsDocLoader(BaseLoader):
    def load_docs_from_file(self, filename):
        loader = UnstructuredFileLoader(filename)
        documents = loader.load()
        return documents

    def clean_document(self, document: str, ext: str) -> str:
        # 替换两个回车符为一个
        document = re.sub(r'\n+', '\n', document).strip()
        # 清除掉页码
        document = re.sub(r'－\d+－', '', document)
        document = re.sub(r'— \d+ —', '', document)
        document = re.sub(r'\d+ -', '', document)
        # 在第XX章、句号前面加一个分块符<chunk>
        pattern = r'(第[\u4e00-\u9fa5\d]{1,20}章|。)'
        replacement = r'<chunk>\1'
        document = re.sub(pattern, replacement, document)
        # 返回结果
        return document

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
        ext = self.filename.split(".")[-1].lower()
        documents = self.load_docs_from_file(self.filename)
        documents[0].page_content = self.clean_document(
            documents[0].page_content, ext)
        return documents

    def load_and_split(self):
        documents = self.load()
        # split document
        text_splitter = CommonchsDocSplitter(
            separator="\n\n",
            chunk_size=512,
            chunk_overlap=100,
            length_function=len,
        )
        return text_splitter.split_documents(documents)


class CommonchsDocSplitter(CharacterTextSplitter):
    def split_text(this, document: str, ext: str) -> list:
        chunks = document.split("<chunk>")
        if ext == "pdf":
            chunks_without_newline = [chunk.replace(
                "\n", "").strip() for chunk in chunks]
        else:
            chunks_without_newline = [chunk.strip() for chunk in chunks]
        return chunks_without_newline

    def __init__(self, separator: str = "\n\n", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._separator = separator

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        filename = _metadatas[0]["source"]
        ext = filename.split(".")[-1].lower()
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text, ext):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)


# python commonchs_splitter.py --file .\documents\其他文件\低算力条件下GPT应用开发2024.1.5.docx
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    # load document
    loader = CommonchsDocLoader(args.file)
    splitdocs = loader.load_and_split()
    for i, doc in enumerate(splitdocs):
        print(f'doc #{i}: {doc.page_content}')
