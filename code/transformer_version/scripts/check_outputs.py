import torch 
import torch.nn.functional as F


loaded_tensor = torch.load("/home/user_7065/project_B/open-unmix-main_modified/scripts/x_100.pt")
print("Loaded true Tensor:")
print(loaded_tensor.shape)
print(loaded_tensor)


loaded_tensor_mine = torch.load("/home/user_7065/project_B/open-unmix-main_modified/scripts/x_encoded_100.pt")
print("Loaded my Tensor:")
print(loaded_tensor_mine.shape)
print(loaded_tensor_mine)

loss = F.mse_loss(loaded_tensor, loaded_tensor_mine)
print("MSE Loss:")
print(loss.item())

diff_tensor=abs(loaded_tensor -loaded_tensor_mine)
print("diff Tensor:")
print(diff_tensor)

# Sort the difference tensor in descending order
sorted_diff, indices = torch.sort(diff_tensor.reshape(-1), descending=True)

# Display the top 20 values
print("Top 20 absolute differences:")
print(sorted_diff[:40])
# Find the original positions of the top 20 differences
tensor_shape = diff_tensor.shape

# Get the original positions of the top 20 differences manually
top_20_indices = indices[:200]
top_20_positions = []
for flat_index in top_20_indices:
    position = []
    for dim in reversed(tensor_shape):
        position.append(flat_index % dim)
        flat_index //= dim
    top_20_positions.append(tuple(reversed(position)))

print("Top 20 absolute differences with indices:")
for i, pos in enumerate(top_20_positions):
    print(f"Value: {sorted_diff[i]}, Position: {pos}")

print(diff_tensor[1,0,69,13])
print(loaded_tensor[1,0,69,13])
print(loaded_tensor_mine[1,0,69,13])