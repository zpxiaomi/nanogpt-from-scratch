# Mathmetical trick
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 3500
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)
# B,T,C = 4,8,2
# x = torch.randn(B,T,C)


# original version of weighted sum
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b,:t+1]
#         xbow[b, t] = torch.mean(xprev, 0) 
# print(x)
# print('x[0]:')
# print(x[0])
# print('xbow[0]:')
# print(xbow[0])

# trick version of weighted sum
# torch.manual_seed(42)
# a = torch.tril(torch.ones(3, 3))
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.randint(0, 10, (3,2)).float()
# c = a @ b
# print('a=')
# print(a)
# print('b=')
# print(b)
# print('c=')
# print(c)


# version 2
# wei = torch.tril(torch.ones(T, T))
# wei = wei / wei.sum(1, keepdim=True)
# xbow2 = wei @ x # (B, T, T) @ (B,T,C)  --> (B, T, C)
# print(torch.allclose(xbow, xbow2, atol=1e-7))  # atol raised: torch.mean vs matmul accumulate differently, ~3e-8 diff

# print('xbow[0]:')
# print(xbow[0])
# print('xbow2[0]:')
# print(xbow2[0])

# version 3 use softmax
# tril = torch.tril(torch.ones(T,T))
# print(tril)

# wei = torch.zeros((T, T))
# wei = wei.masked_fill(tril==0, float('-inf'))
# wei = F.softmax(wei, dim=-1)
# xbow3 = wei @ x
# print(torch.allclose(xbow, xbow3, atol=1e-7))

# version 4 single head attention
# torch.manual_seed(1337)
# B,T,C = 4, 8, 32
# x = torch.randn((B,T,C))

# head_size = 16
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)
# k = key(x) # (B, T , 16)
# q = query(x) # (B, T , 16)
# # affinity between tokens
# wei = q @ k.transpose(-2, -1) # (B, T , 16) * (B, 16, T) -----> (B, T, T)

# tril = torch.tril(torch.ones(T, T))
# wei = wei.masked_fill(tril == 0, float('-inf')) # no allow to communicate with future   
# wei = F.softmax(wei, dim=-1)
# #out = wei @ x
# v = value(x)
# out = wei @ v
# print(out.shape)
# print(wei[0])

# read the input and assign it to var
with open('/root/data/faust_combined.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) 

# create a map of string to postion / position to string
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #take a string, for each character in string, map it to integer and create a list of these integers
decode = lambda l: ''.join(itos[i] for i in l) #take a list of integers, for each integer , map it to a character and joint these characters to string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading method
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # computer attetnion scores: affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by non linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ transformer block : communication + computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.postion_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final Layernorm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # apply multiple blocks
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
    
    # at step 3000, or just save at the end
    if iter == 3000:
        torch.save(model.state_dict(), '/root/checkpoint_step3000.pt')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# k = torch.randn(B, T,  head_size)
# q = torch.randn(B, T,  head_size)
# wei = q @ k.transpose(-2,-1) * head_size**-0.5

# print(k.var())
# print(q.var())
# print(wei.var())