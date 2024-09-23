# 混合 AI 架构下的 Web 应用开发探讨

## 安装 Ollama 框架

- 访问 Ollama 官网 [https://ollama.com/](https://ollama.com/) 下载安装包
- 安装完成后 Ollama 会自启动，检查 [http://localhost:11434/](http://localhost:11434/) 是否启动成功
- 如果 Ollama 没有自启动，则执行 `ollama serve` 命令手动启动 Ollama 服务
- 设置系统环境变量 `OLLAMA_MODELS` 的值为模型保存的路径，比如 `D:\ollama_models`
- 设置系统环境变量 `OLLAMA_ORIGINS` 的值为 `*`，允许跨域访问 Ollama 服务
- 设置完系统环境变量，需要重启 Ollama 服务

## 下载并安装 Llama 模型

- 访问 Hugging Face 官网 [https://huggingface.co/](https://huggingface.co/) 或镜像 [https://hf-mirror.com/](https://hf-mirror.com/) 下载模型
- 用 `llama gguf` 作为关键字过滤模型，下载 Llama 模型的 `.gguf` 文件，选择 `Meta-Llama-3.1-8B-Instruct-GGUF`
- 在 `Files and versions` 下载 [Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf](https://hf-mirror.com/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main) 保存到 `models` 目录下
- 在 `models` 目录下创建或编辑 `llama.modelfile` 文件，内容为 `FROM ./Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`
- 在 `models` 目录下执行创建命令：`ollama create llama3:8b -f ./llama.modelfile`
- 执行 `ollama list` 查看 `llama3:8b` 模型是否在列表中，执行 `ollama run llama3:8b` 测试该模型

## 安装 Chroma 向量数据库

- 访问 Python 官网 [https://www.python.org/](https://www.python.org/) 下载并安装（已安装请忽略）
- 执行安装 Chroma 命令：`pip install chromadb`
- 打开根目录下的 `.env` 文件，添加 `ANONYMIZED_TELEMETRY=False` 禁用 Chroma 发送收集的数据
- 在 `.env` 文件添加 `CHROMA_SERVER_CORS_ALLOW_ORIGINS='["http://localhost:3000","http://127.0.0.1:5500"]'`
- 在根目录下执行 `chroma run` 命令，检查 [http://localhost:8000/](http://localhost:8000/) 是否启动成功
- 在根目录下会自动创建 `chroma_data` 目录，可以访问 [http://localhost:8000/docs](http://localhost:8000/docs) 测试 Chroma 提供的 API

## 安装工程依赖，下载 Gemma 模型

- 在根目录下，执行 `yarn` 命令安装依赖，将安装 LangChain 框架和 Chroma 客户端
- 可以执行 `node rag.js` 命令以测试 Ollama、Chroma、LangChain 是否协同工作
- 执行 `npm run serve` 启动 3000 和 5500 服务端口，通过 [http://localhost:3000/](http://localhost:3000/) 或 [http://127.0.0.1:5500/](http://127.0.0.1:5500/) 访问网站
- 访问 [Google AI Studio](https://aistudio.google.com/app/apikey) 申请 `Gemini API key`，将 key 保存到根目录的 `key.js` 文件里
- 访问 Kaggle 官网 [https://www.kaggle.com/](https://www.kaggle.com/) 下载 [gemma-1.1-2b-it-gpu-int4.bin](https://www.kaggle.com/models/google/gemma/tfLite/gemma-1.1-2b-it-gpu-int4) 保存到 `models` 目录下

## 访问混合 AI 应用示例

- 应用示例使用的模型说明
  - 本地 Gemma `gemma-1.1-2b-it-gpu-int4.bin` 模型
  - 本地 Llama `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` 模型
  - 本地 Gemini Nano 谷歌浏览器内置的 `window.ai` 模型
  - 远程 Gemini `gemini-1.5-flash` 模型（需要 `Gemini API key`）
- 单个应用里混合用本地和远程两个模型，即智能预定酒店混合模型
  - 应用示例网址 [http://localhost:3000/pawa/](http://localhost:3000/pawa/) （远程 Gemini 模型 + 本地 Gemma 模型）
- 多个应用里混合用本地和远程多个模型，即预订机票与预订酒店应用联动
  - IndexedDB 版本（只支持同域应用联动）
    - 机票预定 [http://localhost:3000/indexedDB/flight.html](http://localhost:3000/indexedDB/flight.html)（远程 Gemini 模型）
    - 酒店预定 [http://localhost:3000/indexedDB/hotel.html](http://localhost:3000/indexedDB/hotel.html)（本地 Gemma 模型）
  - ChromaDB 版本（支持跨域应用联动）
    - 机票预定 [http://127.0.0.1:5500/chromaDB/flight.html](http://127.0.0.1:5500/chromaDB/flight.html) （远程 Gemini 模型）
    - 酒店预定 [http://localhost:3000/chromaDB/hotel.html](http://localhost:3000/chromaDB/hotel.html)（本地 Llama 模型）
  - WebAI 版本（使用 IndexedDB 暂不支持跨域）
    - 机票预定 [http://localhost:3000/webAI/flight.html](http://localhost:3000/webAI/flight.html)（远程 Gemini 模型）
    - 酒店预定 [http://localhost:3000/webAI/hotel.html](http://localhost:3000/webAI/hotel.html)（本地 Gemini Nano 模型）
  - UnifiedAI 版本
    - 机票预定 & 酒店预定 [http://localhost:3000/webAI/unified.html](http://localhost:3000/webAI/unified.html)（本地远程混合模型）
