import './App.css';
import React, { useState, useEffect } from 'react';
import '@chatui/core/es/styles/index.less';
import './chatui-theme.css';
import Chat, { Bubble, useMessages, Modal } from '@chatui/core';
import '@chatui/core/dist/index.css';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: '0000', dangerouslyAllowBrowser: true, baseURL: "http://127.0.0.1:8060/v1" });
const wsbase = "ws://127.0.0.1:8060/ws/";

var message_history = [];

function App() {
  const { messages, appendMsg, setTyping, updateMsg } = useMessages([]);
  const [file, setFile] = useState("");
  const [showUpload, setShowUpload] = useState(false)
  const [clientid, setClientid] = useState("");

  /* eslint-disable */
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      var r = Math.random() * 16 | 0,
        v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  useEffect(() => {
    const _clientid = generateUUID();
    setClientid(_clientid);
    const socket = new WebSocket(wsbase + _clientid);
    socket.onmessage = function (event) {
      updateMsg(_clientid, {
        type: "text",
        content: { text: event.data }
      });
    };
  }, []);

  async function chat_stream(prompt, _msgId) {
    message_history.push({ role: 'user', content: prompt });
    const stream = openai.beta.chat.completions.stream({
      model: 'ChatGLM3-6B',
      messages: message_history,
      stream: true,
    });
    var snapshot = "";
    for await (const chunk of stream) {
      snapshot = snapshot + chunk.choices[0]?.delta?.content || '';
      updateMsg(_msgId, {
        type: "text",
        content: { text: snapshot.trim() }
      });
    }
    message_history.push({ "role": "assistant", "content": snapshot });
  }

  const defaultQuickReplies = [
    {
      icon: 'message',
      name: '翻译pdf',
      isNew: false,
      isHighlight: true,
    }
  ];

  function handleSend(type, val) {
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
      });
      const msgID = new Date().getTime();
      setTyping(true);
      appendMsg({
        _id: msgID,
        type: 'text',
        content: { text: '' },
      });
      chat_stream(val, msgID);
    }
  }

  function renderMessageContent(msg) {
    const { content } = msg;
    return <Bubble content={content.text} />;
  }

  function handleFileChange(e) {
    setFile(e.target.files[0]);
  }

  function handleUploadClose() {
    setShowUpload(false);
  }

  function handleUploadFile() {
    setShowUpload(false);
    if (file === undefined || file === "") {
      appendMsg({
        type: 'text',
        content: { text: '未指定文件名！' },
        position: 'left',
      });
    } else {
      appendMsg({
        _id: clientid,
        type: 'text',
        content: { text: '开始上传文件' },
        position: 'left',
      });
      uploadFile();
    }
  }

  function handleQuickReplyClick(item) {
    if (item.name.startsWith("翻译pdf")) {
      setShowUpload(true);
      return;
    }
  }

  async function uploadFile() {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('clientid', clientid);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        console.log('File uploaded successfully');
      } else {
        console.error('File upload failed');
      }
    };
    xhr.send(formData);
  };

  return (
    <div style={{ height: 'calc(100vh - 2px)', marginTop: '-5px' }}>
      <Chat
        navbar={{ title: 'chat-app' }}
        messages={messages}
        renderMessageContent={renderMessageContent}
        quickReplies={defaultQuickReplies}
        onQuickReplyClick={handleQuickReplyClick}
        onSend={handleSend}
      />
      {
        <div>
          <Modal
            active={showUpload}
            title="翻译pdf"
            showClose={false}
            onClose={handleUploadClose}
            actions={[
              {
                label: '上传',
                color: 'primary',
                onClick: handleUploadFile,
              },
              {
                label: '取消',
                onClick: handleUploadClose,
              },
            ]}
          >
            <div style={{ overflow: 'hidden' }}>
              <input type="file" accept=".pdf" onChange={handleFileChange} />
            </div>
          </Modal>
        </div>

      }
    </div>
  );
}

export default App;
