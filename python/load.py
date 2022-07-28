import torch

FILE = "../model_ignore/swin_squished_ranger_jit.pt"
file2 = "../model_ignore/export.pkl"

# model = torch.jit.load(FILE)
# model.eval()

model2 = torch.load(file2)
model2.eval()

