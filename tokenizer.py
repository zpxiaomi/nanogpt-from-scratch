# example of simple enccoding and decoding

# read the story and store text in a variable
with open('./data/faust_combined.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in chracters: ", len(text))

# print the first 1000 characters
#print(text[:1000])

# get all unique characters in the story file
chars = sorted(list(set(text)))
vacab_size = len(chars)
print(''.join(chars))
print(vacab_size) 

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


