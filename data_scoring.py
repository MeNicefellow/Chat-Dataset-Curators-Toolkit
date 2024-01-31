import requests
import random
import json
from datasets import load_dataset, Dataset, DatasetDict
import multiprocessing
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
        f = open("failed_samples.txt", "a",encoding='utf-8')
        f.write(f"Failed to score: {sample}\n-----------\nwith response: {request.json()['choices'][0]['text']}\n")
        f.close()
    return res

def worker(samples):
    return [score_sample(sample, host_add, api_key) for sample in samples]

def score_dataset(dataset, host_add, api_key):
    res = {}
    dataset_dict = load_dataset(dataset)
    so_far = 0
    failed = 0
    num_processes = multiprocessing.cpu_count()

    for split_name, split_dataset in dataset_dict.items():
        res[split_name] = {'text':[], 'score':[], 'rationale':[]}
        print("Split: ", split_name)

        # Split the dataset into chunks
        samples = split_dataset['text']
        chunks = [samples[i::num_processes] for i in range(num_processes)]

        # Create a pool of worker processes
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(worker, chunks)

        # Merge the results
        results = [result for chunk_results in results for result in chunk_results]

        for sample, score_ana in zip(samples, results):
            so_far += 1
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