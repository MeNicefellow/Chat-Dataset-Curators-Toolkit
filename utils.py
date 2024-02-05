import requests
import random

def ask_LLM(prompt, host, api_key,backend = 'tabyapi',inst_beg = "[INST]",inst_end = "[/INST]",max_tokens = 1024,min_p = 0.9,top_k = 1,top_p = 0.9,temperature=0.8):

    if backend == 'vllm':
        payload = {
            "prompt": prompt,
            "model": "/workspace/text-generation-webui2/models/TheBloke_Mistral-7B-Instruct-v0.2-AWQ",
            "max_tokens": max_tokens,
            "min_p": min_p,
            "stream": False,
            "seed": random.randint(
                1000002406736107, 3778562406736107
            ),  # Was acting weird without this
            "top_k": top_k,
            "top_p": top_p,
            "stop": ["</s>", inst_beg, inst_end],
            "temperature": temperature,
        }
    else:
        payload = {
            "prompt": prompt,
            "model": "gpt-3.5-turbo-instruct",
            "max_tokens": max_tokens,
            "n_predict": max_tokens,
            "min_p": min_p,
            "stream": False,
            "seed": random.randint(
                1000002406736107, 3778562406736107
            ),  # Was acting weird without this
            "top_k": top_k,
            "top_p": top_p,
            "stop": ["</s>", inst_beg, inst_end],
            "temperature": temperature,
        }

    request = requests.post(
        host,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=360,
        stream=False,
    )
    return request.json()['choices'][0]['text']