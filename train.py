from torch.optim import Adam
import torch.nn as nn
import torch
from model import StackBiLSTMClassifier
from utils import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import sys

BATCH_SIZE = 32
epochs = 10
glove_path = sys.argv[1]
path_model = 'best_model_800.pth' if '800' in glove_path else 'best_model_300.pth'
MODEL_PATH = f'model/{path_model}'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# some code take from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    cum_loss = 0
    for batch, (prem, hyp, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        prem = (prem[0].to(device), prem[1])
        hyp = (hyp[0].to(device), hyp[1])
        y = y.to(device)
        pred = model(prem, hyp)
        loss = loss_fn(pred, y.squeeze())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss
    return cum_loss/len(dataloader)



def test_loop(dataloader, model, loss_fn, dataset='TEST'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for prem, hyp, y in dataloader:
            prem = (prem[0].to(device), prem[1])
            hyp = (hyp[0].to(device), hyp[1])
            y = y.to(device)
            pred = model(prem, hyp)
            test_loss += loss_fn(pred, y.squeeze()).item()
            correct += (pred.argmax(1) == y.squeeze()).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"{dataset} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

if __name__ == '__main__':

    dataset = load_dataset("snli")
    glove_path = sys.argv[1]  #'glove/glove.6B.300d.txt'
    # glove_path = 'glove/glove.840B.300d.txt'
    words, vecs = read_glove_vector(glove_path)
    w2i = {word: i+1 for i, word in enumerate(words)}

    dataset = dataset.filter(lambda example: example['label'] != -1)

    train_set = dataset['train']
    test_set = dataset['test']
    val_set = dataset['validation']

    train_set = NLIDataset(train_set, w2i)
    test_set = NLIDataset(test_set, w2i)
    val_set = NLIDataset(val_set, w2i)

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)


    # Initialize the loss function
    model = StackBiLSTMClassifier(
        vocab_size=len(words),
        embedding_dim=300
    )
    model.encoder.embeddings.weight.data.copy_(torch.from_numpy(vecs))
    model = model.to(device)
    print(model)

    lr = 0.0002
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.2)
    best_acc = 0
    best_model = model
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        acc, loss = test_loop(val_dataloader, model, loss_fn, dataset='VALIDATION')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
        scheduler.step()
        print(f"Epoch {t + 1}, Validation Accuracy: {acc * 100}, Validation Loss: {loss}, Train loss: {train_loss}")
    print("Done!")
    test_loop(test_dataloader, model, loss_fn)
