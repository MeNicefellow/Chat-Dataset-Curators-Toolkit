
import json
from datasets import load_dataset, Dataset, DatasetDict
import multiprocessing
from tqdm import tqdm
from functools import partial
import os
import pickle
from utils import ask_LLM

def score_sample(sample,host_add,api_key,backend = 'vllm'):

    inst_beg = "[INST]"
    inst_end = "[/INST]"

    instruction = "Please score the following chatbot responses on a scale of 1-5, where 1 is the worst and 5 is the best. Please return the result in json format with two keys: score and rationale.\n\n"
    appended_str = '{"score": '




    prompt = inst_beg + instruction + sample + inst_end + appended_str
    try:
        res = ask_LLM(prompt,host_add,api_key=)
        res = appended_str+res
        res = json.loads(res)
        res['sample'] = sample
    except:
        res = {'sample': sample,'score':0,'rationale':'Failed to score'}
        f = open("failed_samples.txt", "a",encoding='utf-8')
        f.write(f"Failed to score: {sample}\n-----------\nwith response: {request.json()['choices'][0]['text']}\n")
        f.close()
    return res

def worker(sample, host_add, api_key):
    results = []
    results.append(score_sample(sample, host_add, api_key))
    return results

def score_dataset(dataset, host_add, api_key):
    res = {}
    nam = dataset.split('/')[1]
    intermediate_results = f'intermediate_results_{nam}.pkl'
    if os.path.exists(intermediate_results):
        with open(intermediate_results, 'rb') as f:
            res = pickle.load(f)

    dataset_dict = load_dataset(dataset)
    so_far = 0
    failed = 0
    num_processes = 1#multiprocessing.cpu_count()//2
    chunk_size = 500

    for split_name, split_dataset in dataset_dict.items():
        if split_name not in res:
            res[split_name] = {'text':[], 'score':[], 'rationale':[]}
        print("Split: ", split_name)

        # Determine how many samples have already been scored
        num_scored_samples = len(res[split_name]['text'])

        # Calculate the number of chunks to skip
        num_chunks_to_skip = num_scored_samples // chunk_size

        # Split the dataset into chunks
        samples = split_dataset['text']
        chunks = [samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)]

        # Skip the chunks of already scored samples
        chunks = chunks[num_chunks_to_skip:]

        # Create a pool of worker processes
        with multiprocessing.Pool(num_processes) as pool:
            # Create a partial function with fixed arguments
            worker_func = partial(worker, host_add=host_add, api_key=api_key)

            # Use tqdm to show progress
            for chunk in tqdm(chunks):
                results = pool.map(worker_func, chunk)

                # Merge the results
                results = [result for chunk_results in results for result in chunk_results]

                for score_ana in results:
                    so_far += 1
                    if score_ana['score'] == 0:
                        failed += 1
                        print(f"Failure rate so far: {failed/so_far:.2%}")
                    res[split_name]['text'].append(score_ana['sample'])
                    res[split_name]['score'].append(score_ana['score'])
                    res[split_name]['rationale'].append(score_ana['rationale'])

                # Save intermediate results after each chunk
                with open(intermediate_results, 'wb') as f:
                    pickle.dump(res, f)

    for split_name, split_dataset in res.items():
        res[split_name] = Dataset.from_dict(split_dataset)

    result_dataset_dict = DatasetDict(res)
    return result_dataset_dict