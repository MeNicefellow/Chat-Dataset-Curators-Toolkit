from datasets import load_dataset, Dataset, DatasetDict


def convert_Alpaca(dataset,sys_prefix,user_prefix,assist_prefix):
    res = []
    for sample in dataset:
        tmp = ''
        if sample['input'] == '':
            tmp += sys_prefix+user_prefix+sample['instruction']
        else:
            tmp += sample['instruction']+user_prefix+sample['input']
        tmp += assist_prefix+sample['output']
        res.append(tmp)
    return res


def convert_sharegpt(dataset,sys_prefix,user_prefix,assist_prefix):
    res = []
    for sample in dataset:
        tmp = sys_prefix
        for talk in sample['conversations']:
            if talk['role'] == 'user':
                tmp += user_prefix+talk['value']
            else:
                tmp += assist_prefix+talk['value']

        res.append(tmp)
    return res

def convert_datasets(original_datasets,sys_prefix,user_prefix,assist_prefix):
    res = {}
    for dataset in original_datasets:
            dataset_dict = load_dataset(dataset[1])
            for split_name, split_dataset in dataset_dict.items():
                if split_name not in res:
                    res[split_name] = []
                    if dataset[0] == 'alpaca':
                        res[split_name].extend(convert_Alpaca(split_dataset,sys_prefix,user_prefix,assist_prefix))
                    elif dataset[0] == 'sharegpt':
                        res[split_name].extend(convert_sharegpt(split_dataset,sys_prefix,user_prefix,assist_prefix))
    for split_name, split_dataset in res.items():
        res[split_name] = Dataset.from_dict({"text": split_dataset})
    dataset_dict = DatasetDict(res)
    return dataset_dict


if __name__ == '__main__':

    original_datasets = [('alpaca',"yahma/alpaca-cleaned"), ('sharegpt',"erfanzar/ShareGPT4")]


    sys_prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    user_prefix = "\n\nUSER: "
    assist_prefix = "\nASSISTANT: "
    convert_datasets(original_datasets, sys_prefix, user_prefix, assist_prefix)

