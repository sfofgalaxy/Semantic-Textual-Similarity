from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate
from openprompt import PromptForClassification
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
import torch
import pandas as pd
from transformers import  AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,BertTokenizer,Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("--form", default="prefix")
parser.add_argument("--re", default='f', help="resume or not")
parser.add_argument("--t", default='f', help="test or not")
parser.add_argument("--ep", default='5', help="test or not")

args = parser.parse_args()
FORM = args.form
checkpoint_dir = FORM+'_prompt_best_model.pt'

def load_data(filepath):
    df = pd.read_csv(filepath)
    ret = []
    for index, data in df.iterrows():
        input_example = InputExample(text_a = data['sentence1'], text_b = data['sentence2'], label=int(data['label']), guid=index)
        ret.append(input_example)
    return ret

dataset = {}
dataset['train'] = load_data('./data/train.csv')
dataset['dev'] = load_data('./data/dev.csv')
dataset['test'] = load_data('./data/test.csv')

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "hfl/chinese-roberta-wwm-ext")

promptTemplate = {
    'suffix':ManualTemplate(
        text = '{"placeholder":"text_a"}。{"placeholder":"text_b"}。上面两句话{"mask"}相似。',
        tokenizer = tokenizer), 
    'prefix':ManualTemplate(
        text = '下面两句话{"mask"}相似。{"placeholder":"text_a"}。{"placeholder":"text_b"}。',
        tokenizer = tokenizer),
    'soft_prefix':MixedTemplate(
        text = '{"soft"}{"soft"}{"soft"}{"soft"}{"mask"}{"soft"}{"soft"}。{"placeholder":"text_a"}。{"placeholder":"text_b"}。',
        model=plm, tokenizer = tokenizer),
    'soft_suffix':MixedTemplate(
        text = '{"placeholder":"text_a"}。{"placeholder":"text_b"}。{"soft"}{"soft"}{"soft"}{"soft"}{"mask"}{"soft"}{"soft"}。',
        model=plm, tokenizer=tokenizer)
}

# tokenizer = BertTokenizer(vocab_file='./data/vocab.txt')
wrapped_tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
model_inputs = {}
for split in ['train', 'dev', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_tokenizer.tokenize_one_example(promptTemplate[FORM].wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)


promptVerbalizer = ManualVerbalizer(
    num_classes= 2,
    label_words = [
        ["不"],
        ["很"]
    ],
    tokenizer = tokenizer,
)

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=promptTemplate[FORM], tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=32,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# Evaluate
validation_dataloader = PromptDataLoader(dataset=dataset["dev"], template=promptTemplate[FORM], tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

# start training
use_cuda = True
if args.re == 'f' and args.t == 'f':
    prompt_model = PromptForClassification(plm=plm,template=promptTemplate[FORM], verbalizer=promptVerbalizer, freeze_plm=False)
else:
    prompt_model = torch.load(checkpoint_dir)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# Now the training is standard
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, weight_decay=0.01)
acc = 0
length = len(train_dataloader)

if args.t == 'f':
    for epoch in range(int(args.ep)):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step %1000 == 1:
                print("Epoch {}, step: {}/{}, average loss: {}".format(epoch, step, length, tot_loss/(step+1)), flush=True)
        
                # Evaluation
                allpreds = []
                alllabels = []
                for step, inputs in enumerate(validation_dataloader):
                    if use_cuda:
                        inputs = inputs.cuda()
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    alllabels.extend(labels.cpu().tolist())
                    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                report = classification_report(alllabels, allpreds, digits=4)
                print(report)

                cur_acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
                if (cur_acc>acc):
                    acc = cur_acc
                    torch.save(prompt_model,FORM+'_prompt_best_model.pt')




# Test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=promptTemplate[FORM], tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

allpreds = []
alllabels = []
prompt_model = torch.load(checkpoint_dir)
for step, inputs in enumerate(test_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    print(logits)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

cm = confusion_matrix(y_true=alllabels, y_pred=allpreds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./result/open_prompt+'+FORM+'.png')#保存图片
report = classification_report(alllabels, allpreds, digits=4)
my_open = open('./result/open_prompt+'+FORM+'.txt', 'w')
my_open.write(report)
my_open.close()

print(report)