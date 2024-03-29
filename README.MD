#  Chat Dataset Curators' Toolkit


## Overview
The Unified Chat Dataset Converter is a tool designed to standardize various instruction/chat datasets into a single, unified format and then filter it with LLM-based chatbots. This project aims to facilitate research and development in Large Language Models by providing a consistent structure for diverse datasets.

Instruction following and chat datasets can both be seen as a system information followed by n rounds of user and assistant speeches. However, different formats are used for different datasets, which makes it difficult to train on them with a uniform format. The Unified Chat Dataset Converter in the toolkit aims to solve this problem by converting various datasets into a unified format.



## Features
- **Prefix Customization:** Users can specify prefixes for system information, user inputs, and AI replies, ensuring compatibility with different processing models.
- **Broad Compatibility:** Offers functions to convert various original datasets available on Huggingface to the unified format.
- **Easy Integration:** Designed for straightforward integration with existing NLP pipelines and workflows.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- Access to datasets on Huggingface

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/MilesQLi/Unified-Chat-Dataset-Converter.git
   ```
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage of Unified Chat Dataset Converter
1. Set the prefixes for system information, user input, and AI replies in the configuration file.
2. Define a list of the datasets you want to convert from Huggingface, including their types and paths.


   Example:
   ```python
    from Dataset_unifier import convert_datasets

    sys_prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    user_prefix = "\n\nUSER: "
    assist_prefix = "\nASSISTANT: "
    original_datasets = [('alpaca','teknium/GPTeacher-General-Instruct','input','response','instruction'),('alpaca','cognitivecomputations/dolphin,flan1m-alpaca-uncensored'),('alpaca','truthful_qa',None,'best_answer','question'),('sharegpt', "erfanzar/ShareGPT4")]

    convert_datasets(original_datasets, sys_prefix, user_prefix, assist_prefix)
   ```

Please check the details in the `original_datasets` variable to see how to handle different situations.

The aforementioned example will convert the datasets into Vicuna-1.1 format. The converted datasets will be combined to a DatasetDict and returned.


### Supported Datasets
List of types of datasets from Huggingface that are currently supported by this tool:
- Alpaca
- ShareGPT
- Capybara
- ...


## License
Apache 2.0

## TODO
- [ ] Add support for more dataset types
- [ ] Add support for dataset cleaning