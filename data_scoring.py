import requests
import random
import json
from datasets import load_dataset, Dataset, DatasetDict

from tqdm import tqdm


def score_sample(sample,host_add,api_key):
    max_tokens = 1024
    min_p = 0.9
    top_k = 1
    top_p = 0.9
    temperature=0.8

    inst_beg = "[INST]"
    inst_end = "[/INST]"

    instruction = "Please score the following chatbot responses on a scale of 1-5, where 1 is the worst and 5 is the best. Please return the result in json format with two keys: score and rationale.\n\n"
    appended_str = '{"score": '




    prompt = inst_beg + instruction + sample + inst_end + appended_str

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
        host_add,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=360,
        stream=False,
    )
    try:
        res = request.json()['choices'][0]['text']
        res = appended_str+res
        res = json.loads(res)
    except:
        res = {'score':0,'rationale':'Failed to score'}
        print("============\nFailed to score: ",sample,'\n-----------\nwith response: ',request.json()['choices'][0]['text'])
    return res

def score_dataset(dataset,host_add,api_key):
    res = {}
    dataset_dict = load_dataset(dataset)
    so_far = 0
    failed = 0
    for split_name, split_dataset in dataset_dict.items():
        res[split_name] = {'text':[],'score':[],'rationale':[]}
        print("Split: ",split_name)
        for sample in tqdm(split_dataset['text']):
            so_far += 1
            score_ana = score_sample(sample,host_add,api_key)
            if score_ana['score'] == 0:
                failed += 1
                print(f"Failure rate so far: {failed/so_far:.2%}")
            res[split_name]['text'].append(sample)
            res[split_name]['score'].append(score_ana['score'])
            res[split_name]['rationale'].append(score_ana['rationale'])
    for split_name, split_dataset in res.items():
        res[split_name] = Dataset.from_dict(split_dataset)

    result_dataset_dict = DatasetDict(res)
    return result_dataset_dict