# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from model import TextLSTM

def tokenizer(word):
    return [letter for letter in word]
    
if __name__ == "__main__":
    TEXT = data.Field(sequential=True, tokenize=tokenizer)
    train = data.TabularDataset(
        path="train.csv",
        format="csv",
        fields=[("text", TEXT)]
    )

    TEXT.build_vocab(train)

    hidden_size = 128
    n_step = 3
    input_size = len(TEXT.vocab)

    train_iter = data.BucketIterator(
        dataset=train,
        batch_size=3,
        repeat=False
    )

    model = TextLSTM(input_size, hidden_size)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        for batch in train_iter:
            input_batch = batch.text[:-1]
            target_batch = batch.text[-1]

            optimizer.zero_grad()
            hidden = model.init_hidden()

            output = model(input_batch, hidden)

            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

        if(epoch + 1) % 500 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    




    
