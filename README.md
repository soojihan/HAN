# Introduction

This repository contains source code for our paper:
Sooji, Han, Rui Mao, and Erik Cambria. "Hierarchical Attention Network for Explainable Depression Detection on Twitter Aided by Metaphor Concept Mappings." In Proceedings of the 29th International Conference on Computational Linguistics (COLING), 2022. in press

# Datasets
We are currently working on the release of the datasets (tweets and associated metaphor concept mappings publicly.
In the meantime, please contact [Sooji Han](suji.han.x@gmail.com) for requesting the datasets.

# How to run

### Training


Key inputs for training are as follows:
1) train_set_path ("-t"): train dataset path
2) validation_set_path("-d"): validation dataset path
3) evaluationset("-e"): evaluation (test) dataset
4) model_file_prefix("-p"): model file prefix  for model weight output file
5) --post_dir: a directory where tweet json files are stored
6) --max_post_size: maximum social context size (default 200)
7) n_gpu ("-g"): gpu device(s) to use (-1: no gpu, 0: 1 gpu). only support int value for device no.
8) epochs: set num_epochs for training

1. Download a pre-trained language model (e.g. bert-base-uncased; https://huggingface.co/bert-base-uncased/tree/main)
2. In ```depression_classifier.py```, change paths for the following two parameters accordingly:<br/>
        ```self.embedding_tokenizer``` = AutoTokenizer.from_pretrained('[your_dir]/bert-base-uncased')<br/>
        ```self.embedding_model``` = AutoModel.from_pretrained('[your_dir]/bert-base-uncased').to(self.cuda_device)<br/>
        <br/>The dimensions of tweets and metaphor concept mappings embeddings are set to 768 by default as 'bert-base-uncased' was used in our experiments. If another pre-trained language model is used, the dimension parameter should be modified in ```depression_classifier.py``` accordingly. 


3. Run a command:<br/>
Example usage:<br/>
python src/model_torch/trainer.py -t '[your_dir]/train.csv' -d '[your_dir]/dev.csv' -e '[your_dir]/test.csv' --post_dir '[your_dir]/mddl_metaphor_input' -p 'training_test' -g 0 --epochs 10 --max_post_size 200

# Contact
[Sooji Han](https://soojihan.github.io/)<br/>
[Rui Mao](https://maorui.wixsite.com/homepage)
