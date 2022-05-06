import argparse
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast, RobertaTokenizerFast
import binary_model

import train
import dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--mod', default='bt',
                        help='use different pretrained model to train and test')
    parser.add_argument('--ep', default=10,
                        help='assign the number of epochs')
    parser.add_argument('--test', default='f',
                        help='the model is test or not')
    parser.add_argument('--pre', default='f',
                        help='use pretrained tokenizer or not')
    parser.add_argument('--bs', default='128',
                        help='batch size')
    parser.add_argument('--restore', default='f', help='restore or not')
    args = parser.parse_args()

    if args.mod == 'bt':
        model_name = 'bert-base-chinese'
        backbone = AutoModel.from_pretrained(model_name)
        tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    elif args.mod == 'rb':
        model_name = 'hfl/chinese-roberta-wwm-ext'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)
    elif args.mod =='hflrb':
        model_name = "hfl/chinese-roberta-wwm-ext"
        backbone = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif args.mod == 'simbt':
        model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
        backbone = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif args.mod == 'simrb':
        model_name = "princeton-nlp/sup-simcse-roberta-base"
        backbone = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else :
        model_name = args.mod
        backbone = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.pre == 't':
        tokenizer = BertTokenizer.from_pretrained('./data/vocab.txt') 
    
    data = dataloader.my_data('./data', tokenizer, int(args.bs))
    model = binary_model.create_model(backbone, tokenizer)
    device = torch.device('cuda:1')
    
    if args.test == 'f':
        if args.restore == 't':
            path = './saved_model/'+model._get_name()+'.pt'
            model.load_state_dict(torch.load(path))
        train.training(model, data, device, int(args.ep))
    # test
    train.testing(model, data, device)