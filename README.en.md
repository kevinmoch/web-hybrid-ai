# Exploring Web Application Development with a Hybrid AI Architecture

## Installing the Ollama Framework

- Download the installation package from the Ollama official website: [https://ollama.com/](https://ollama.com/)
- After installation, Ollama will start automatically. Check if it's running successfully at [http://localhost:11434/](http://localhost:11434/)
- If Ollama doesn't start automatically, run the command `ollama serve` to start the Ollama service manually.
- Set the system environment variable `OLLAMA_MODELS` to the path where your models are saved, e.g., `D:\ollama_models`
- Set the system environment variable `OLLAMA_ORIGINS` to `*` to allow cross-origin access to the Ollama service.
- After setting the environment variables, restart the Ollama service.

## Downloading and Installing the Llama Model

- Download the model from Hugging Face: [https://huggingface.co/](https://huggingface.co/) or its mirror: [https://hf-mirror.com/](https://hf-mirror.com/)
- Filter models using the keyword `llama gguf` , download the `.gguf` file for the Llama model, and choose `Meta-Llama-3.1-8B-Instruct-GGUF`
- In `Files and versions` , download [Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf](https://hf-mirror.com/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main) and save it to the `models` directory.
- In the `models` directory, create or edit the `llama.modelfile` with the content: `FROM ./Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`
- In the `models` directory, execute the creation command: `ollama create llama3:8b -f ./llama.modelfile`
- Run `ollama list` to check if the `llama3:8b` model is in the list, and then run `ollama run llama3:8b` to test the model.

## Installing Chroma Vector Database

- Download and install Python from [https://www.python.org/](https://www.python.org/) (ignore if already installed).
- Install Chroma using the command: `pip install chromadb`
- Open the `.env` file at the root directory and add `ANONYMIZED_TELEMETRY=False` to disable Chroma data collection.
- Add `CHROMA_SERVER_CORS_ALLOW_ORIGINS='["http://localhost:3000","http://127.0.0.1:5500"]'` to the `.env` file.
- Run the `chroma run` command in the root directory and check if it's running successfully at [http://localhost:8000/](http://localhost:8000/)
- A `chroma_data` directory will be automatically created in the root directory. You can test the Chroma API at [http://localhost:8000/docs](http://localhost:8000/docs)

## Installing Project Dependencies and Downloading the Gemma Model

- In the root directory, run the `yarn` command to install dependencies, including the LangChain framework and Chroma client.
- You can test the integration of Ollama, Chroma, and LangChain by running `node rag.js`
- Start the 3000 and 5500 service ports by running `npm run serve` . Access the website at [http://localhost:3000/](http://localhost:3000/) or [http://127.0.0.1:5500/](http://127.0.0.1:5500/)
- Apply for a `Gemini API key` at [Google AI Studio](https://aistudio.google.com/app/apikey) and save it to the `key.js` file in the root directory.
- Download [gemma-1.1-2b-it-gpu-int4.bin](https://www.kaggle.com/models/google/gemma/tfLite/gemma-1.1-2b-it-gpu-int4) from Kaggle: [https://www.kaggle.com/](https://www.kaggle.com/) and save it to the `models` directory.

## Accessing Hybrid AI Application Examples

- Model Overview:
  - Local Gemma model: `gemma-1.1-2b-it-gpu-int4.bin`
  - Local Llama model: `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`
  - Local Gemini Nano model: built-in `window.ai` in Google Chrome
  - Remote Gemini model: `gemini-1.5-flash` (requires Gemini API key)
- Single Application with Hybrid Models: Intelligent hotel booking hybrid model using a local and remote model.
  - Application example: [http://localhost:3000/pawa/](http://localhost:3000/pawa/) (remote Gemini model + local Gemma model)
- Multiple Applications with Hybrid Models: Flight and hotel booking applications interacting with each other.
  - IndexedDB version (only supports same-domain application interaction):
    - Flight booking: [http://localhost:3000/indexedDB/flight.html](http://localhost:3000/indexedDB/flight.html) (remote Gemini model)
    - Hotel booking: [http://localhost:3000/indexedDB/hotel.html](http://localhost:3000/indexedDB/hotel.html) (local Gemma model)
  - ChromaDB version (supports cross-domain application interaction):
    - Flight booking: [http://127.0.0.1:5500/chromaDB/flight.html](http://127.0.0.1:5500/chromaDB/flight.html) (remote Gemini model)
    - Hotel booking: [http://localhost:3000/chromaDB/hotel.html](http://localhost:3000/chromaDB/hotel.html) (local Llama model)
  - WebAI version (uses IndexedDB, doesn't support cross-domain):
    - Flight booking: [http://localhost:3000/webAI/flight.html](http://localhost:3000/webAI/flight.html) (remote Gemini model)
    - Hotel booking: [http://localhost:3000/webAI/hotel.html](http://localhost:3000/webAI/hotel.html) (local Gemini Nano model)
  - UnifiedAI version:
    - Flight booking & Hotel booking: [http://localhost:3000/webAI/unified.html](http://localhost:3000/webAI/unified.html) (hybrid local and remote models)
