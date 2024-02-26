from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
import os
import argparse
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = None
tokenizer = None
base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)
html_url = "http://127.0.0.1:8060/download/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trans_model():
    global model, tokenizer
    modelpath = "./dataroot/models/Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelpath).to(device)


def translate_text(text):
    tokenized_text = tokenizer(
        text, return_tensors='pt', truncation=True, padding=True).to(device)
    tokenized_text['repetition_penalty'] = 2.85
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(
        translation, skip_special_tokens=True)
    return translated_text[0]


def translate_text_llm(text):
    messages = [
        {
            "role": "user",
            "content": "请翻译下列英文，直接给出结果，无法翻译的部分直接原样输出:" + text
        }
    ]
    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        stream=False,
        max_tokens=2048,
        temperature=1,
        presence_penalty=1.1,
        top_p=0.8)
    if response:
        content = response.choices[0].message.content
        return content
    else:
        print("Error:", response.status_code)
        return text


def translate_html(pdf, html, llm, clientid, fn):
    print("开始翻译html")
    with open('output.html', 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    divs = soup.find_all('div', class_="t")
    pbar = tqdm(total=len(divs))
    lines = []
    special = [2, 17]
    for div in divs:
        pbar.update(1)
        if fn is not None:
            pg = round(pbar.n / pbar.total * 100, 2)
            fn(clientid, "已完成：" + str(pg) + "%")
        # 用div的class属性判断跳过哪些div
        skip_flag = False
        for x in range(20, 51):
            if "m" + str(x) in div["class"]:
                skip_flag = True
                break
        for sp in special:
            if "ff" + str(sp) in div["class"]:
                skip_flag = True
                break
        if skip_flag:
            continue
        # 取div内完整文字，忽略span
        text = div.get_text(strip=True, separator=' ')
        # 翻译div
        if text is not None and text != "" and len(text) > 5:
            if llm:
                _text = translate_text_llm(text)
                if _text == "(逗号)":
                    _text = _text.replace("(逗号)", "")
                    if "翻译" in _text or "原文" in _text or "无法" in _text or "拼写错误" in _text or "直译" in _text or "这段英文" in _text:
                        _text = translate_text(text)
            else:
                _text = translate_text(text)
            div.string = _text
            lines.append(text + "====" + _text + "\n")
    with open(html, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    with open('log.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("翻译完成，生成文件：" + html)
    fn(clientid, "下载链接：" + html_url + clientid + ".html")


def pdf2html(pdf, clientid, fn):
    if fn is not None:
        fn(clientid, "开始pdf转换到html")
    if not os.path.exists(pdf):
        return False
    outfile = "output.html"
    if os.path.exists(outfile):
        os.remove(outfile)
    cmd = "pdf2htmlEX --zoom 1.5 " + pdf + " " + outfile
    os.system(cmd)
    if fn is not None:
        fn(clientid, "开始翻译html")
    return os.path.exists(outfile)


def translate_pdf_html(pdf, html, llm, clientid, fn):
    if model is None:
        load_trans_model()
    if pdf2html(pdf, clientid, fn):
        translate_html(pdf, html, llm, clientid, fn)
        return True
    else:
        return False


if __name__ == "__main__":
    print("torch cuda:", torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default=None, type=str, required=True)
    parser.add_argument('--html', default="translated.html",
                        type=str, required=False)
    parser.add_argument('--llm', action='store_true',
                        default=False, required=False)
    args = parser.parse_args()
    translate_pdf_html(args.pdf, args.html, args.llm, None, None)
