from tkinter import *
from numpy import place
from transformers import Trainer,AutoModelForSequenceClassification,BertTokenizer
from datasets import Dataset
import tkinter.font as tkFont

tokenizer = BertTokenizer(vocab_file='./data/vocab.txt')
model = AutoModelForSequenceClassification.from_pretrained('./result/checkpoint-29856/')
def create_trainer():
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )
    return trainer

root = Tk()
root.title('Calculate Semantic Textual Similarity')
root.geometry("400x250")
fontStyle = tkFont.Font(family="Lucida Grande", size=15)

Label(root, text='Sentence 1:', font=fontStyle).grid(row=0, column=0)
Label(root, text='Sentence 2:', font=fontStyle).grid(row=1, column=0)

 
e1 = Entry(root)
e2 = Entry(root)
e1.grid(row=0, column=2, padx=0, pady=10,columnspan=3)
e2.grid(row=1, column=2, padx=0, pady=10,columnspan=3)

result = StringVar()
result.set('Result: ')
l = Label(root,textvariable=result, font=fontStyle)
l.grid(row=3, column=0, padx=0, pady=20)

trainer = create_trainer()
def show():  #当输入内容时点击获取信息会打印
    sentence1 = e1.get()
    sentence2 = e2.get()

    text = sentence1 + '[SEP]' + sentence2
    di = {'text': [text], 'labels': [0]}
    dataset = Dataset.from_dict(di)
    def preprocess_function(examples):
        text_token = tokenizer(examples['text'], truncation = True, max_length = 128)
        return text_token
    dataset = dataset.map(preprocess_function, batched=True)

    raw_pred = trainer.predict(dataset)
    result.set('Result: '+ ('Not ' if raw_pred.predictions[0].argmax(-1)==0 else '') +'Similar')
    
Button(root, text='Calculate', width=10, command=show).grid(row=2, column=1, sticky=W, padx=10, pady=5)

root.mainloop()

