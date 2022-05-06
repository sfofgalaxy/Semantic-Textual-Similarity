import torch.nn as nn

######################################
# create a model for classification  #
######################################

class model_arch(nn.Module):
    def __init__(self, backbone):
        super(model_arch, self).__init__()
        self.backbone = backbone 
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer class_num (Output layer)
        self.fc2 = nn.Linear(512, 2)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        _, cls_hs = self.backbone(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        #x dim 512
        x = self.relu(x)
        # x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x

def create_model(backbone, tokenizer):
    # import BERT-base pretrained model
    backbone.resize_token_embeddings(len(tokenizer))
    # freeze all the parameters
    for param in backbone.parameters():
        param.requires_grad = False
    # pass the pre-trained BERT to our define architecture
    model = model_arch(backbone)
    return model

