Here's the English translation of your text:

1. The frontend framework used is **open-webui**; customized modifications can be found in the directory **open-webui-modify**.

2. LoRA fine-tuning was done using **LlammaFactory**, with specific adjustments for table tennis knowledge; personalized data is stored in the directory **LLama-factory**.

3. **ollama** is used as the container for local models, as shown in the figure.    ![image](https://github.com/user-attachments/assets/dcca227b-4406-439b-9334-4c8b4f0734ef)

4. The LLM invokes a CV model via function calling to obtain the raw data. The statistical data from the CV model is then provided back to the LLM, integrated, and returned as part of the conversation. Refer to the detailed setup process below：https://drive.google.com/file/d/1vDDk703_aPP-gEuTbifo_vJGzbbuUqpf/view?usp=sharing
