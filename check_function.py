import torch
batch_size = 10
# Assuming X and Y are your tensors
X_shape = (batch_size, 1000, 500)
Y_shape = (batch_size, 1, 100)

# Create random tensors for demonstration
X = torch.rand(X_shape)
Y = torch.rand(Y_shape)

# Broadcast Y to match the shape of X along the second dimension
Y_broadcasted = Y.expand(-1, 1000, -1)

# Concatenate X and Y along the third dimension
concatenated_tensor = torch.cat((X, Y_broadcasted), dim=2)

# Check the shape of the concatenated tensor
print(concatenated_tensor.shape)