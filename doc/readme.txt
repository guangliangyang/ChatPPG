1. open-webui
    https://github.com/open-webui/open-webui
    https://docs.openwebui.com/
    https://docs.openwebui.com/getting-started/quick-start/


    GPU & AUTH
    docker run -d -p 3000:8080 --gpus all -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda

2. Ollama
    https://ollama.com/
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ollama pull llama3.2
    ollama list
    ollama serve (already running)
    ollama run llama3.2
