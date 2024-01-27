from Dataset_unifier import convert_datasets

sys_prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
user_prefix = "\n\nUSER: "
assist_prefix = "\nASSISTANT: "
original_datasets = [('alpaca','teknium/GPTeacher-General-Instruct','input','response','instruction'),('alpaca','cognitivecomputations/dolphin,flan1m-alpaca-uncensored'),('alpaca','truthful_qa',None,'best_answer','question'),('sharegpt', "erfanzar/ShareGPT4")]

dataset = convert_datasets(original_datasets, sys_prefix, user_prefix, assist_prefix)