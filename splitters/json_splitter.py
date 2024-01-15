import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import argparse

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        meta_key: Optional[str] = None,
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._meta_key = meta_key

    def create_documents(self, json_data):
        documents = []
        for item in json_data:
            content = ''.join(item[self._content_key])
            document = Document(page_content=content, metadata={
                                "source": item[self._meta_key]})
            documents.append(document)
        return documents

    def load(self) -> List[Document]:
        docs = []
        with open(self.file_path, 'r') as json_file:
            try:
                json_data = json.load(json_file)
                docs = self.create_documents(json_data)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format")
        return docs

    def load_and_split(self):
        return self.load()


# python splitters/json_splitter.py --file ./data/output.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    # load document
    loader = JSONLoader(args.file, "q", "a")
    splitdocs = loader.load_and_split()
    for i, doc in enumerate(splitdocs):
        print(f'doc #{i}: {doc.page_content}')
        if i > 10:
            break
