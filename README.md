# QueryChain查询链（知识库+大模型）

QueryChain是基于langchain的本地知识库+大语言模型的参考实现，客户端发出查询请求时，先查询知识库，从Chroma向量库中查询出词条，然后再将词条发送到大语言模型归纳生成答案。

- 知识库：基于langchain的生成与检索

- Chain：查询知识库+调用大模型

- 目录监控：将原始文档复制到documents文件夹，程序感知后加入知识库

- 流式输出：从知识库的查询结果与从大模型生成的结果以流方式逐步推送到客户端

- OpenAI兼容接口：标准/v1/chat/completions接口，客户端使用标准OpenAI接口调用

- 文本分隔器自动适配：根据文档内容选择不同的文本分隔算法，如政策文件、一般中文和英文算法

- pdf翻译：使用Helsinki-NLP/opus-mt-en-zh翻译模型和通用大语言模型翻译英文pdf文件

  ![](https://gitclone.com/download1/aiit/qchain.gif)

## 1、服务器端环境安装

```bash
# （1）新建python环境
conda create -n qchain python=3.10 -y
conda activate qchain 
# （2）安装依赖包
pip install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com
# （3）如果在Windows上，默认装的是cpu版的torch，需要重装成GPU版的
# 验证
在python提示符下，运行
import torch
print(torch.__version__)
print(torch.cuda.is_available())
# 如果是CPU版的，需要从网站上找到命令安装
pip uninstall torch -y 
pip uninstall torchvision -y
# cuda 12.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# cuda 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com
# （4） 下载文本转向量模型
python tools/model_download.py --repo_id shibing624/text2vec-base-chinese
```

## 2、URL配置

### （1）服务器端

```bash
toos/pdf_trans.py有两处：
1、base_url：指向大语言模型的服务地址，如http://后台机器IP:8000/v1/
2、html_url：客户端下载翻译后的html地址：如http://后台机器IP:8060/download/
config.py
openid_base_url:指向大语言模型的服务地址，如http://后台机器IP:8000/v1
```

### （2）客户端

```
chat-app/src/app.js两处：
1、知识库后台openai兼容接口的baseURL：如前后台在一台机器上设为http://127.0.0.1:8060/v1，前后端分离设为http://后台机器IP:8060/v1
2、知识库后台websocket地址：如前后台在一台机器上设为http://127.0.0.1:8060/ws/，前后端分离设为http://后台机器IP:8060/ws/
```

## 3、运行服务器端

```shell
# 首次运行程序要注意以下两个问题
# （1） 程序会下载tokenizers/punkt，时间比较长
# 如果时间太长，可尝试以下方法：
# python
# >>> import nltk
# >>> nltk.download('punkt')
# >>> exit()
# （2） 需要安装jvm
# .docx文件解析器需要jvm，比如在unbutu上，sudo apt-get install default-jdk
# 方式一：正常运行，装载已生成的向量库（库文件在indexs目录下）
conda activate qchain
python qchain.py
# 方式二：重建向量库，删除indexs目录，重新生成向量库
python qchain.py --reindex
# Linux上后台运行
pkill -f -9 qchain
nohup python -u qchain.py > qchain.log 2>&1 &
tail -f qchain.log
# 如果用到pdf翻译功能，将tools/pdf2htmlEX.rar解压到querychain的根目录
```

## 4、运行客户端

```shell
cd chat-app
npm i
npm start
```

## 5、测试客户端

http://localhost:3000

## 6、知识库操作

### （1）增加知识库文件

将docx、txt、md、pdf、json等文件复制到documents目录下即可，程序会监控文件夹变化，定时将新文件内容分块存入知识库。

### （2）JSON格式文件说明

文件整理成以下格式，也可以不是q和a节点，在代码中loader = JSONLoader(filename,"q","a")修改。这样写入知识库里用q做Embedding，用a做metadata，检索时用q匹配问题，用a显示答案。

```json
[
    {
        "q":"问题1？",
        "a":"答案1。"
    },
    {
        "q":"问题2？",
        "a":"答案2。"
    }
]
```

### （3）重新生成知识库

python qchain.py --reindex

## 7、API调用说明

使用OpenAI兼容接口/v1/chat/completions调用，是一种sse（Server-Send Events，服务器端推流）模式，以下代码适用于node.js、react.js、vue等，其他语言可参照openai的接口调用SDK说明。

使用前先用npm i opanai安装依赖包

```javascript
import OpenAI from 'openai';

//创建OpenAI客户端
const openai = new OpenAI({ apiKey: '0000', dangerouslyAllowBrowser: true, baseURL: "http://127.0.0.1:8060/v1" });
//全局变量，每次调用chat_stream的入参与返参都保存到message_history，下次再调用时携带上次的问答历史
//也可以不用全局变量，而是chat_stream时做为入参将message_history传入也可
var message_history = [];

async function chat_stream(prompt) {
    message_history.push({ role: 'user', content: prompt });
    const stream = openai.beta.chat.completions.stream({
      model: 'ChatGLM3-6B',
      messages: message_history,
      stream: true,
    });
    var snapshot = "";
    for await (const chunk of stream) {
      snapshot = snapshot + chunk.choices[0]?.delta?.content || '';
      //增量生成内容
      console.log(chunk.choices[0]?.delta?.content || '') ;
      //全部生成内容
      console.log(snapshot.trim()) ;
    }
    message_history.push({ "role": "assistant", "content": snapshot });
}
```

## 8、nginx sse配置

如果API服务使用了nginx代理，会出现nginx缓冲sse流的问题，导致的现象是客户端调用sse接口时，流数据并不是按流式一条一条出现的，而是最后一次性返回，原因是nginx网关对流数据进行了缓存，可按以下配置解决，如果经过了多道nginx转发，则每个nginx都需要配置。

```bash
location /v1 {
  proxy_http_version 1.1;
  proxy_set_header Connection "";
  proxy_set_header X-Real-IP $remote_addr;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  proxy_read_timeout 86400s;
  proxy_buffering off;
  proxy_cache off;
  proxy_pass http://127.0.0.1:8060/v1/ ;
}
```

