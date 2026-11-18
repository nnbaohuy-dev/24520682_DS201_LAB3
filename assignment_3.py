import os 
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from tqdm import tqdm
import numpy as np

from data_utils.PhoNer import PhoNer, Vocab, collate_fn
from models.encoder.encoder import Encoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):

    running_loss = []
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"].to(device)
            
            # forward pass
            logits = model(input_ids, lengths)      # (bs, n_labels)
            logits = logits.transpose(1, 2) 

            loss = loss_fn(logits, labels) 
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({
                "loss": np.array(running_loss).mean()
            })
    return float(np.mean(running_loss))

@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> float:
    model.eval()

    gts = []        # ground truths 
    pds = []        # predictions 
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"].to(device) # (bs, )
            
            # forward pass
            logits: torch.Tensor = model(input_ids, lengths)
            predicted_labels = logits.argmax(dim=-1) # (bs, )

            for pred, label in zip(predicted_labels, labels):
                valid_mask = label != -1

                pred = pred[valid_mask]
                label = label[valid_mask]

                pds.extend(pred.cpu().tolist())
                gts.extend(label.cpu().tolist())
                
    f1 = f1_score(gts, pds, average="macro")
    print(f"F1: {f1}")
    return f1

if __name__ == "__main__":
    
    print("Loading vocab ...")
    vocab = Vocab(train_path="datasets/Phoner/train_word.json")

    print("Creating datasets ...")
    train_dataset = PhoNer(
        path="datasets/Phoner/train_word.json",
        vocab=vocab
    )
    dev_dataset = PhoNer(
        path="datasets/Phoner/dev_word.json",
        vocab=vocab
    )
    test_dataset = PhoNer(
        path="datasets/Phoner/test_word.json",
        vocab=vocab
    )

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = 32,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size = 32,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size = 32,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Initializing model and utilities ...")
    model = Encoder(
        vocab_size=vocab.len,
        hidden_size=256,
        n_layer=5,
        n_labels=vocab.n_labels,
        padding_idx=0
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    epoch = 0
    best_f1 = 0
    patience = 0
    while True:
        epoch += 1
        train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch)

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
        else:
            patience += 1

        if patience == 5:
            print("Patience reached! Training completed")
            break
    
    test_f1 = evaluate(model, test_dataloader, epoch)
    print(f"F1-score on test set: {test_f1}")
