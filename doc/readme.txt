
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

    add my_pipeline (just for logging)

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
        You answer to the user DIRECTLY as the tool answers,show all JSON data  from provided tool in a table by data subsection .
        not from yourself, not combine the old context, only the latest answer from provided tool
    ---------------------------------------------------------------------------

    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------


5. Tools (most important)
    add my_tool.py

6. update dataloader, in docker file, and restart docker container
     open-webui\backend\open_webui\apps\retrieval\loaders\main.py

7. pipeline
    my_pipeline.py


8. model prompt suggests

    Summarize performance, highlight weaknesses, and suggest areas for improvement.
     detect serve foul, and provide targeted corrective measures

     questions:
    Create two-week plan for player, improve consistency and recovery.
     Suggest specific exercises to enhance player’s stability and control.
     Design multiball drills to improve reaction speed and footwork.

10. prompt

1) please analysis the player performance

2) Summarize performance, highlight weaknesses, and suggest areas for improvement

3) Determine the skill level of the player based on the above data. Give me the exact numerical reason and proof or evidence

4) please detect serve foul



