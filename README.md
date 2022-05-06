# Acknowledgement
This is code repository of PENG, Zifan in HKUST Course CSIT 6910A Independent Project, which can be seen on [GitHub Code repository](https://github.com/sfofgalaxy/Semantic-Texual-Similarity).

The basic project task is to detect the similarity of two given sentences, output the similarity of the two sentences, similar or not.

# Data
Dataset is in the `data` folder. The dataset has been pre-split into three parts, training, development and test datasets. Which is from this paper, called [LCQMC:A Large-scale Chinese Question Matching Corpus](https://aclanthology.org/C18-1166/).


# Result
The result can be found in folder `result`, which shown the confusion matrix and result report of every model.

# Demo

`demo.py` is the visuailized demo which shows the similarity of two given sentences. But `demo.py` depends on the checkpoint which I will released on Google Drive later.

# Models

## Model1
`model1` folder contains the original version model, which get a not good result and with inappropriate hyper-parameters. To train it, just run the command in the root path:
```.bash
# run the command in the Semantic-Texual-Similarity with epochs of 20
python model1/train.py --ep 20
```

## Model2
`model2` folder contains the better model with [huggingface](http://huggingface.co/) trainer version model, it get a better results with better optimizer and hyper-parameters. And also the contrastive learning models, but this task is not with the pre-trained chinese vocabulary, so the results are not seems good. To train it, just run the command in the root path:
```.bash
# run the command in the Semantic-Texual-Similarity with roberta model
python model2/init.py --ep 20 --type rb
```

## Prompt
`prompt` folder contains the better model with [OpenPrompt](https://github.com/thunlp/OpenPrompt/) trainer version model, it get a better results with Prompt-tuning which is very popular currently, which is based on `huggingface` pre-trained models. and hyper-parameters. I trained it with different prompt templates. To train it with your own template, you need to change the source code and just run the command in the root path:
```.bash
# run the command in the Semantic-Texual-Similarity with roberta model
python prompt/open_prompt.py
```

