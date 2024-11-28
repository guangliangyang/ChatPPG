
1. Ollama
    https://ollama.com/
    CPU
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    GPU
    docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

    ollama pull llama3.2
    ollama list
    ollama serve (already running)
    ollama run llama3.2

2. open-webui
    https://github.com/open-webui/open-webui
    https://docs.openwebui.com/
    https://docs.openwebui.com/getting-started/quick-start/

    docker volume rm open-webui

    GPU & AUTH
    docker run -d -p 3000:8080 --gpus all -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda

3. pipelines

    docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main
    dmin Panel > Settings > Connections ->Manage OpenAI API Connections
    http://host.docker.internal:9099  0p3n-w3bu!

4. video loader
    in open-webui docker
    apt-get update
    apt-get install -y libmagic1 libmagic-dev

5. you can access uploaded files by adding an inlet function, if you upload a file, you should see it in the body:

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)

        return body

    async def inlet(self, body: dict, user: dict) -> dict:
    print(f"Received body: {body}")
    files = body.get("files", [])
    for file in files:
        content_url = file["url"] + "/content"
        print(f"file available at {content_url}")
        # read the file content as binary and do something ...
    return body

6. prompt

 You will get a picture from the user or ask a user politely for a picture  to analysis. Once you get the picture,  user may ask you 2 kinds of question:
1. analysis player performance(analy_table_tennis_perfomance in  provided tool )
2. detect serve foul (detect_serve_foul in  provided tool )
all of these user request, you MUST use a provided tool to calculate.
You answer to the user DIRECTLY as the tool answers, not from yourself


 please analysis the player performance
 please detect serve foul


