# QueryChain查询链（知识库+大模型）

QueryChain是基于langchain的本地知识库+大语言模型的参考实现，客户端发出查询请求时，先查询知识库，从Chroma向量库中查询出词条，然后再将词条发送到大语言模型归纳生成答案。

- 知识库：基于langchain的生成与检索
- Chain：查询知识库+调用大模型
- 目录监控：将原始文档复制到documents文件夹，程序感知后加入知识库
- 流式输出：从知识库的查询结果与从大模型生成的结果以流方式逐步推送到客户端
- OpenAI兼容接口：标准/v1/chat/completions接口，客户端使用标准OpenAI接口调用
- 文本分隔器自动适配：根据文档内容选择不同的文本分隔算法，如政策文件、一般中文和英文算法

## 1、服务器端环境安装

```bash
conda create -n qchain python=3.10 -y
conda activate qchain 
# 安装依赖包
pip install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com
# 下载文本转向量模型
python tools/model_download.py --repo_id shibing624/text2vec-base-chinese
```

## 2、运行服务器端

```shell
conda activate qchain
# 方式一：正常运行，装载已生成的向量库（库文件在indexs目录下）
python qchain.py
# 方式二：重建向量库，删除indexs目录，重新生成向量库
python qchain.py --reindex
# Linux上后台运行
nohup python -u qchain.py > qchain.log 2>&1 &
tail -f qchain.log
```

## 3、运行客户端

```shell
cd chat-app
npm i
npm start
```

## 4、测试客户端

http://localhost:3000

## 5、知识库操作

### （1）增加知识库文件

将docx、txt、md、pdf等文件复制到documents目录下即可，程序会监控文件夹变化，定时将新文件内容分块存入知识库。

### （2）重新生成知识库

python qchain.py --reindex

## 6、API调用说明

使用OpenAI兼容接口/v1/chat/completions调用，以下代码适用于node.js、react.js、vue等，其他语言可参照openai的接口调用SDK说明。

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

