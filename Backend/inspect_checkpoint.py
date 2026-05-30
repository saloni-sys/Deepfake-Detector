import torch

checkpoint = torch.load(
    "weights/efficientnet_b4_best.pth",
    map_location="cpu"
)

print(checkpoint.keys())

state_dict = checkpoint["state_dict"]

print("Number of keys:", len(state_dict))
print("First 20 keys:")

for i, key in enumerate(state_dict.keys()):
    print(key)
    if i == 19:
        break