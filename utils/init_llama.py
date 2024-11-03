# utils/init_llama.py

import requests
from langchain_core.messages import HumanMessage, SystemMessage

class LLM:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def invoke(self, messages):

        serializable_messages = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, SystemMessage)):
                serializable_messages.append({"role": msg.__class__.__name__, "content": msg.content})
            else:
                raise TypeError(f'Unsupported message type: {type(msg)}')

        response = requests.post(self.endpoint, json={"messages": serializable_messages})
        response.raise_for_status()
        return response.json()

# Crear una instancia de LLM
llm_json_mode = LLM('http://127.0.0.1:1234/v1/models/')
