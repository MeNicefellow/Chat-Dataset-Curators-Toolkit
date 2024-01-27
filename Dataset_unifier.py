from datasets import load_dataset, Dataset, DatasetDict


def convert_Alpaca(dataset,sys_prefix,user_prefix,assist_prefix,inp_col='input',out_col='output',inst_col='instruction'):
    res = []
    for sample in dataset:
        tmp = ''
        if inp_col == None or sample[inp_col] == '':
            tmp += sys_prefix+user_prefix+sample[inst_col]
        else:
            tmp += sample[inst_col]+user_prefix+sample[inp_col]
        tmp += assist_prefix+sample[out_col]
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


def convert_capybara(dataset,sys_prefix,user_prefix,assist_prefix):
    res = []
    for sample in dataset:
        tmp = sys_prefix
        for talk in sample['conversation']:
            tmp += user_prefix+talk['input'] + assist_prefix+talk['output']
        res.append(tmp)
    return res

def convert_datasets(original_datasets,sys_prefix,user_prefix,assist_prefix):
    res = {}
    i = 0
    n_datasets = len(original_datasets)
    for dataset in original_datasets:
        i += 1
        print(f"Processing dataset {i}/{n_datasets}: {dataset[1]}")
        if ',' in dataset[1]:
            name,branch = dataset[1].split(',')
            dataset_dict = load_dataset(name,branch)
        else:
            dataset_dict = load_dataset(dataset[1])
        for split_name, split_dataset in dataset_dict.items():
            if split_name not in res:
                res[split_name] = []
            if dataset[0] == 'alpaca':
                if len(dataset) == 2:
                    new_dataset = convert_Alpaca(split_dataset,sys_prefix,user_prefix,assist_prefix)
                else:
                    assert(len(dataset) == 5)
                    inp_col = dataset[2]
                    out_col = dataset[3]
                    inst_col = dataset[4]
                    new_dataset = convert_Alpaca(split_dataset,sys_prefix,user_prefix,assist_prefix,inp_col,out_col,inst_col)
            elif dataset[0] == 'sharegpt':
                new_dataset = convert_sharegpt(split_dataset,sys_prefix,user_prefix,assist_prefix)
            elif dataset[0] == 'capybara':
                new_dataset = convert_capybara(split_dataset,sys_prefix,user_prefix,assist_prefix)
            res[split_name].extend(new_dataset)
            print(f"Split {split_name} added {len(new_dataset)} samples")
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

