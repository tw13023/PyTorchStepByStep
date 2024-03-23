torch.manual_seed(13)
# build tensors from numpy arrays BEFORE creating the dataset
x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()

# build dataset containing all data points 
dataset = TensorDataset(x_tensor, y_tensor)

# Perform the split 
ratio = 0.8
n_total = len(dataset)
n_train = int(ratio * n_total)
n_val = n_total - n_train 
train_data, val_data = random_split(dataset, [n_train, n_val])

# Build a loader for each set 
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)
