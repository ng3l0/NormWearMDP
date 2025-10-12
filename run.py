import torch
from main_model import NormWearModel

# config
device = torch.device('cpu')

### init model ##################################################################
weight_path = "normwear_last_checkpoint-15470-correct.pth"
model = NormWearModel(weight_path=weight_path, optimized_cwt=True).to(device)

# generate data
# test example: 2 samples, 3 sensor, sequence length of 2 seconds
# data shape notation: bn for batch size, nvar for number of sensor channels,
# P for number of patches, E for embedding dimension (768)
sampling_rate = 64
x = torch.rand(2, 3, sampling_rate*2).to(device)

# encoding
out = model.get_embedding(x, sampling_rate=sampling_rate, device=device) # bn, nvar, P, E

# log
print("Input shape:", x.shape) # [2, 3, 128]
print("Output shape:", out.shape) # [2, 3, P, 768]