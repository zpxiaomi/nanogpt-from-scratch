import torch
from model import GPTLanguageModel, device, decode


model = GPTLanguageModel()
model.load_state_dict(torch.load('checkpoint_step3000.pt'))
model = model.to(device=device)
model.eval()

# generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))