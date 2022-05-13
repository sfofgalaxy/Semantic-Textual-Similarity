#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, BertTokenizer, BertForMaskedLM, AdamW, Trainer, TrainingArguments, RobertaTokenizer, RobertaForMaskedLM
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
appendix_pattern = {1:['']}
logging.basicConfig(level=logging.INFO)
answer = {0: '不', 1:'很'}
text = '下面两句话{"mask"}相似。',

class LecCallTag():
    # 数据处理
    def data_process(self, data_file):
        df = pd.read_csv(data_file)

        train_data = np.array(df)
        train_data_list = train_data.tolist()
        for pair in train_data_list:
            pair[1] = '下面两句话' + answer[pair[2]] + '相似。' + pair[0] + pair[1]
            pair[0] = '下面两句话[MASK]相似。' + pair[0] + pair[1]

        df = pd.DataFrame(train_data_list)

        text = df[0].tolist()
        label = df[1].tolist()
        return text, label
    
    # model, tokenizer
    def create_model_tokenizer(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)

        return tokenizer, model
    
    # 构建dataset
    def create_dataset(self, text, label, tokenizer, max_len):
        dic = {'text': text, 'labels': label}
        dataset = Dataset.from_dict(dic)
        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding='max_length',truncation=True, max_length=max_len)
            return text_token
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset
    
    # 构建trainer
    def create_trainer(self, model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer):
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            weight_decay=0.01,
            metric_for_best_model='accuracy',
            load_best_model_at_end = True
        )
        def compute_metrics(pred):
            labels = pred.label_ids[:, 6]
            preds = pred.predictions[:, 6].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            report = classification_report(labels, preds, digits=4)
            logging.info(report)
            
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        return trainer

def test(test_dataset, trainer):
    # Make prediction
    raw_pred = trainer.predict(test_dataset)
    labels = raw_pred.label_ids[:, 6]
    preds = raw_pred.predictions[:, 6].argmax(-1)

    cm = confusion_matrix(y_true=labels, y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('./results/prompt.png')#保存图片

    report = classification_report(labels, preds, digits=4)
    my_open = open('./results/prompt.txt', 'w')
    my_open.write(report)
    my_open.close()

def main(model_name, epoch):
    lct = LecCallTag()
    checkpoint_dir = "./results/checkpoint/"
    batch_size = 32
    max_len = 120
    tokenizer, model = lct.create_model_tokenizer(model_name)

    text, label = lct.data_process('./data/train.csv')
    train_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    
    text, label = lct.data_process('./data/dev.csv')
    val_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer)
    trainer.train()
    text, label = lct.data_process('./data/test.csv')
    test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    test(test_dataset, trainer)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--type', default='bt',
                    help='use different pretrained model to train and test')
    parser.add_argument('--ep', default='10', help='number of epochs')
    args = parser.parse_args()

    if args.type == 'bt':
        model_name = 'bert-base-chinese'
    elif args.type == 'rb':
        model_name = 'hfl/chinese-roberta-wwm-ext'
    elif args.type == 'hflbt':
        model_name = 'hfl/chinese-bert-wwm-ext'

    main(args.type, int(args.ep))
