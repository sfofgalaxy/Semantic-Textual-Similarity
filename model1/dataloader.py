from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import torch



class my_data():
    def __init__(self, path: str, tokenizer, batch_size=128):
        self.path = path
        self.tokenizer = tokenizer

        train_df = pd.read_csv(path + "/train.csv")
        dev_df = pd.read_csv(path + "/dev.csv")
        test_df = pd.read_csv(path + "/test.csv")

        
        train_list= [tuple(row[0:2]) for row in train_df.values]
        dev_list= [tuple(row[0:2]) for row in dev_df.values]
        test_list= [tuple(row[0:2]) for row in test_df.values]
        
        tokens_train, tokens_dev, tokens_test = self.load_tokens(train_list, dev_list, test_list)

        self.train_y = train_df['label'].to_list()

        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_label = torch.tensor(self.train_y)

        # for validation set
        dev_seq = torch.tensor(tokens_dev['input_ids'])
        dev_mask = torch.tensor(tokens_dev['attention_mask'])
        dev_label = torch.tensor(dev_df['label'].to_list())

        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        test_label = torch.tensor(test_df['label'].to_list())

        self.train_seq = train_seq
        self.train_mask = train_mask
        self.train_label = train_label
        self.dev_seq = dev_seq
        self.dev_mask = dev_mask
        self.dev_label = dev_label

        train_dataloader = self.craete_dataloader(train_seq, train_mask, train_label, random=True, batch_size=batch_size)
        dev_dataloader = self.craete_dataloader(dev_seq, dev_mask, dev_label, batch_size=batch_size)
        test_dataloader = self.craete_dataloader(test_seq, test_mask, test_label, batch_size=batch_size)

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        
        self.test_seq = test_seq
        self.test_mask = test_mask
        self.test_label = test_label

    ######################################
    # get tokens from original text data #
    ######################################
    def load_tokens(self, df_train_text, df_dev_text, df_test_text):
        # Split train dataset into train, validation and test sets
        # tokenize and encode sequences in the training set
        tokens_train = self.tokenizer.batch_encode_plus(
            df_train_text,
            max_length = 128,
            padding = 'max_length',
            return_attention_mask=True,
            add_special_tokens=True, 
            truncation=True,
            return_token_type_ids=False
        )

        # tokenize and encode sequences in the validation set
        tokens_dev = self.tokenizer.batch_encode_plus(
            df_dev_text,
            max_length = 128,
            padding = 'max_length',
            return_attention_mask=True,
            add_special_tokens=True, 
            truncation=True,
            return_token_type_ids=False
        )

        # tokenize and encode sequences in the test set
        tokens_test = self.tokenizer.batch_encode_plus(
            df_test_text,
            max_length = 128,
            padding = 'max_length',
            return_attention_mask=True,
            add_special_tokens=True, 
            truncation=True,
            return_token_type_ids=False
        )

        return tokens_train, tokens_dev, tokens_test

    ######################################
    # create dataloader from tokens      #
    ######################################
    def craete_dataloader(self, seq, mask, label, random = False, batch_size = 32):
        # wrap tensors
        data = TensorDataset(seq, mask, label)
        # sampler for sampling the data during training
        if random:
            sampler = RandomSampler(data)
        else :
            sampler = SequentialSampler(data)
        # dataLoader for train set
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader