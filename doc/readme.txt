
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


    docker volume rm pipelines
    docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main
    dmin Panel > Settings > Connections ->Manage OpenAI API Connections
    http://host.docker.internal:9099  0p3n-w3bu!

    add my_pipeline

4. model setting
    base model: llama3.2
    description: Analyze table tennis players' performance and give professional suggestions for improvement
    system prompt:

    ----------------------------------------------------------------------
 You will get a picture from the user or ask a user politely for a picture  to analysis. Once you get the picture,  user may ask you 2 kinds of question:
1. analysis player performance(analy_table_tennis_perfomance in  provided tool )
2. detect serve foul (detect_serve_foul in  provided tool )
all of these user request, you MUST use a provided tool to calculate.
You answer to the user DIRECTLY as the tool answers, not from yourself
    ----------------------------------------------------------------------
     You will get a file from the user or ask a user politely for a picture  to analysis. Once you get the file,  user may ask you 2 kinds of question:
        1. analysis player performance(analy_table_tennis_performance in  provided tool )
        2. detect player foul (detect_player_foul in  provided tool )
        all of these user request, you MUST use a provided tool to  perform the computations.
        You answer to the user DIRECTLY as the tool answers, not from yourself, not combine the old context, only the latest answer from provided tool

    ---------------------------------------------------------------------------
    You will receive a file from the user or politely request the user to provide a file (video or picture) for analysis. If the file provided is a video, you must call show_video to display it as an embedded player.

The user may ask you two kinds of questions:

1.Analyze player performance: Use the analy_table_tennis_performance function from the provided tool.
2.Detect player foul: Use the detect_player_foul function from the provided tool.
For all user requests:
1.You MUST use the provided tool to perform the computations.
2.If the message contains a video file, call show_video to display the video alongside your response.
3. Your response to the user MUST match the tool's output directly. Do not provide answers from yourself, do not combine the response with old context, and only use the latest answer returned by the provided tool.

    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------


5. Tools
    add my_tool.py

6. update dataloader, in docker file, and restart docker container
     open-webui\backend\open_webui\apps\retrieval\loaders\main.py

7. pipeline
    my_pipeline.py



10. prompt



 please analysis the player performance
 please detect serve foul


