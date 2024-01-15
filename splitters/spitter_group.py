from langchain_community.document_loaders import UnstructuredFileLoader
import re


def calculate_chinese_percentage(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    total_chars = len(text)
    chinese_percentage = (len(chinese_chars) / total_chars) * 100
    return chinese_percentage


def getSpitterGroup(filename):
    if ".json" in filename.lower():
        return "JSON"
    loader = UnstructuredFileLoader(filename)
    documents = loader.load()
    content = documents[0].page_content
    if ("第一条" in content) and ("第二条" in content) and ("第三条" in content):
        return 'POLICY'
    else:
        if calculate_chinese_percentage(content) > 10.0:
            return 'COMMON_CHS'
        else:
            return 'COMMON'


if __name__ == "__main__":
    x = calculate_chinese_percentage("中文abc中344141文")
    print(x)
    print(x > 10.0)
    print(getSpitterGroup(".\documents\相关法律法规\农产品产地冷链物流服务规范.docx"))
