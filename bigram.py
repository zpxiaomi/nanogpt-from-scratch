# example of simple enccoding and decoding

# read the story and store text in a variable
with open('./data/faust_combined.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in chracters: ", len(text))

# print the first 1000 characters
#print(text[:1000])

# get all unique characters in the story file
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size) 

# use a simple mapping as encoding /decoding Alg, string to position or vice versa

# create a map of string to postion / position to string
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #take a string, for each character in string, map it to integer and create a list of these integers
decode = lambda l: ''.join(itos[i] for i in l) #take a list of integers, for each integer , map it to a character and joint these characters to string

print(encode("Hallo Schätz"))
print(decode(encode("Hallo Schätz")))

# store the text in a torch tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

# Let's separate our dataset into train and validation sets
n = int(0.9*len(data)) # first 90% will be train and rest will be validation
train_data = data[:n]
val_data = data[n:]

# only put block of data to transformers for training because it is expensive
block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print context and target, show the autoregressive training process input => output
    print(f"when input is {context} the target is {target}") 


# produce batches of traning data
torch.manual_seed(1337)
batch_size = 4 # define how many sequences running in parallel
block_size = 8 # define how large is the context of one sequence??

def get_batch(split):
    # generate a small batch data of inputs x and targets y
    input_data = train_data if split == "train" else val_data
    ix = torch.randint(len(input_data) - block_size, (batch_size,))
    x = torch.stack([input_data[i:i+block_size] for i in ix])
    y = torch.stack([input_data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('----------')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context} the target is {target}")


import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # create a table of size vocab_size * vocab_size, each token directly reads off logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers. We make prediction about next token based on a individual identity of a single token
        logits = self.token_embedding_table(idx) #(B, T, C), B is batch_size, T is time = block_size , C is channel = vocab_size

        if targets is None:
            loss = None
        else:
            # calculate the loss
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus on last time step
            logits = logits[:,-1,:] # becomes (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print the prediction shape
#print(logits.shape)
# print(loss)

idx = torch.zeros((1,1), dtype = torch.long)


# create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

for steps in range(15000):
    # sample a batch of data 
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))