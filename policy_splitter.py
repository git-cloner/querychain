from typing import List, Optional, Any, Iterable
import argparse
import re
import copy
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


class PolicyDocLoader(BaseLoader):
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
        # 在第XX章、第XX条、第XX节前面加一个分块符<chunk>
        if ext == "pdf":
            pattern = r'(第[\u4e00-\u9fa5\d]{1,20}章 |第[\u4e00-\u9fa5\d]{1,20}条 |第[\u4e00-\u9fa5\d]{1,20}节 )'
        else:
            pattern = r'(第[\u4e00-\u9fa5\d]{1,20}章[\u3000]|第[\u4e00-\u9fa5\d]{1,20}条[\u3000]|第[\u4e00-\u9fa5\d]{1,20}节[\u3000])'
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


class PolicyDocSplitter(CharacterTextSplitter):
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


# python policy_splitter.py --file .\data\相关法律法规\2.中华人民共和国食品安全法.docx
# python policy_splitter.py --file .\data\相关法律法规\食品添加剂生产企业卫生规范.docx
# python policy_splitter.py --file .\data\相关法律法规\市场监管行业标准管理办法.pdf
# python policy_splitter.py --file .\data\相关法律法规\中华人民共和国反垄断法.pdf
# python policy_splitter.py --file .\data\相关法律法规\中华人民共和国价格管理条例.pdf
# python policy_splitter.py --file .\data\相关法律法规\中华人民共和国市场主体登记管理条例.docx
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    # load document
    loader = PolicyDocLoader(args.file)
    documents = loader.load()
    # split document
    text_splitter = PolicyDocSplitter(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
    )
    splitdocs = text_splitter.split_documents(documents)
    for i, doc in enumerate(splitdocs):
        print(f'doc #{i}: {doc.page_content}')
