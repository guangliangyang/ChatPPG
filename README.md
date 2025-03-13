1. 前端框架使用了open-webui； 个性化修改在目录open-webui-modify
2. ora微调使用了LlammaFactory, 做了一点针对乒乓球知识的LoRa； 个性化数据在目录LLama-factory
3. 使用ollamma作为本地模型容器；如图    ![image](https://github.com/user-attachments/assets/dcca227b-4406-439b-9334-4c8b4f0734ef)

5. 在 LLM用function calling 调CV model来获取原始数据,CV统计数据交给LLM，然后集成在一起，返回对话。具体设置流程参考：https://drive.google.com/file/d/1vDDk703_aPP-gEuTbifo_vJGzbbuUqpf/view?usp=sharing
