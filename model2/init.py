import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,BertTokenizer,Trainer, TrainingArguments
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class LecCallTag():
    # 数据处理
    def data_process(self, data_file):
        df = pd.read_csv(data_file)

        train_data = np.array(df)
        train_data_list = train_data.tolist()
        for pair in train_data_list:
            pair[0] = pair[0] + '[SEP]' + pair[1]

        df = pd.DataFrame(train_data_list)

        text = df[0].tolist()
        label = df[2].tolist()
        return text, label
    
    # model, tokenizer
    def create_model_tokenizer(self, model_name):
        tokenizer = BertTokenizer(vocab_file='./data/vocab.txt')
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    
    # 构建dataset
    def create_dataset(self, text, label, tokenizer):
        dict = {'text': text, 'labels': label}
        dataset = Dataset.from_dict(dict)
        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], truncation = True, max_length = 128)
            return text_token
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset
        
    # 构建trainer
    def create_trainer(self, model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer):
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
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
            labels = pred.label_ids[:]
            preds = pred.predictions[:].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            report = classification_report(labels, preds, digits=4)
            logging.info(report)
            
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        return trainer


def test(test_dataset, trainer, model_name):
    # Make prediction
    raw_pred = trainer.predict(test_dataset)
    labels = raw_pred.label_ids[:]
    preds = raw_pred.predictions[:].argmax(-1)

    cm = confusion_matrix(y_true=labels, y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('./result/'+model_name.replace('/','-')+'.png')#保存图片

    report = classification_report(labels, preds, digits=4)
    my_open = open('./result/'+model_name.replace('/','-')+'.txt', 'w')
    my_open.write(report)
    my_open.close()

def main(model_name, epoch, t):
    lct = LecCallTag()
    checkpoint_dir = "./result/checkpoint/"
    batch_size = 32
    tokenizer, model = lct.create_model_tokenizer(model_name)

    data_file = './data/train.csv'
    text, label = lct.data_process(data_file)
    train_dataset = lct.create_dataset(text, label, tokenizer)
    
    data_file = './data/dev.csv'
    text, label = lct.data_process(data_file)
    val_dataset = lct.create_dataset(text, label, tokenizer)
    
    trainer = lct.create_trainer(model, train_dataset, val_dataset, checkpoint_dir, epoch, batch_size, tokenizer)
    if t == 'f':
        trainer.train()

    data_file = './data/test.csv'
    text, label = lct.data_process(data_file)
    test_dataset = lct.create_dataset(text, label, tokenizer)
    test(test_dataset, trainer, model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--type', default='bt',
                    help='use different pretrained model to train and test')
    parser.add_argument('--ep', default='10', help='number of epochs')
    parser.add_argument('--test', default='f', help='number of epochs')
    args = parser.parse_args()

    if args.type == 'bt':
        model_name = 'bert-base-chinese'
    elif args.type == 'rb':
        model_name = 'hfl/chinese-roberta-wwm-ext'
    elif args.type == 'hflbt':
        model_name = 'hfl/chinese-bert-wwm-ext'
    elif args.type == 'simcse':
        model_name = "slider/simcse-chinese-roberta-wwm-ext" #对比学习模型
    elif args.type == 'sim-bt':
        model_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
    elif args.type == 'sim-rb':
        model_name = 'princeton-nlp/sup-simcse-roberta-large'
    elif args.type == 'sim-rb-b':
        model_name = 'princeton-nlp/sup-simcse-roberta-base'
    # elif args.type == 'dp':
    #     model_name = 'uer/roberta-base-finetuned-dianping-chinese' # 大众点评
    # elif args.type == 'jd':
    #     model_name = 'uer/roberta-base-finetuned-jd-full-chinese' # user reviews from 京东
    # elif args.type == 'news':
    #     model_name = 'uer/roberta-base-finetuned-chinanews-chinese' # 中国新闻网
    # elif args.type == 'if':
    #     model_name = 'uer/roberta-base-finetuned-ifeng-chinese' # 凤凰网
    # elif args.type == 'ele':
    #     model_name = 'hfl/chinese-electra-180g-base-discriminator' #微软压缩参数模型
    else :
        model_name = args.type

    main(model_name, int(args.ep), args.test)