import numpy as np
import torch
from numpy import dtype

from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from neural_network.Transformer_old import GPT

batch_size = 12
seq_length = 64
n_layer = 12
n_head = 12
n_embd = 768
learning_rate = 6e-4  # adam optimizer
max_iters = 600000  # total number of training iterations
beta1 = 0.9
beta2 = 0.95
nr_batch_for_eval = 2 # 200

eval_interval = 1 # 2000
validation_set_len = 3 * seq_length * batch_size
bias = False # do we use bias inside LayerNorm and Linear layers?


def get_data():
    """
    returns book as ind for 'train' and 'val'
    :return: dict book_as_ind for 'train' and 'val'
    """
    book = DataLoader.load_data()
    book_chars = sorted(set(book))
    data_converter = DataConverter(book_chars)
    book_as_ind_all = data_converter.chars_to_ind(book)
    return {'train': book_as_ind_all[validation_set_len:], 'val': book_as_ind_all[:validation_set_len]}


def get_random_batch(train_or_val):
    """
    creates a random batch of size batch_size x seq_length
    :param train_or_val: either 'train' or 'val'
    :return:
    """
    ix = torch.randint(len(book_as_ind[train_or_val]) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((book_as_ind[train_or_val][i:i + seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((book_as_ind[train_or_val][i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    estimates loss as mean of loss of nr_batch_for_eval batches
    :return: losses for 'train' and 'val'
    """
    model.eval()
    losses = {'train': torch.zeros(nr_batch_for_eval), 'val': torch.zeros(nr_batch_for_eval)}
    for k in range(nr_batch_for_eval):
        for state in ['train', 'val']:
            X, Y = get_random_batch(state)
            logits, loss = model(X, Y)
            losses[state][k] = loss.item()
    losses['train'] = losses['train'].mean()
    losses['val'] = losses['val'].mean()
    return losses


# training loop
def train():
    iter_num = 0  # number of iterations in the lifetime of this process
    while iter_num <= max_iters:
        # evaluate the loss on train/val sets
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            X, Y = get_random_batch('train')
            logits, loss = model(X, Y)
            scaler.scale(loss).backward()
        # todo grad_clip
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # flush the gradients as soon as we can, no need for this memory anymore
        iter_num += 1


book_as_ind = get_data()  # dict with 'train' and 'val'
vocab_size = len(set(book_as_ind['train']).union(book_as_ind['val']))
model = GPT(vocab_size, n_embd, seq_length, n_layer, bias, n_head)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(learning_rate, (beta1, beta2))
model = torch.compile(model)
train()
