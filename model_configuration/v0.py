device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# setting learning rate , "eta"
lr = 0.1 
torch.manual_seed(42)

# Now create a model and send it to device 
model = nn.Sequential(nn.Linear(1,1)).to(device)

# define optimizer 
optimizer = optim.SGD(model.parameters(), lr=lr)

# define MES loss function
loss_fn = nn.MSELoss(reduction='mean')
