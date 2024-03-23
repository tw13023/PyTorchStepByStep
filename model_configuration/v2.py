# set the device 
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
# set the learning rate, the 'eta' , and the seed(42)
lr = 0.1 
torch.manual_seed(42)

# creat a model and send it to the device at once 
model = nn.Sequential(nn.Linear(1,1)).to(device)

# define the optimizer, SGD to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# define a MSE loss function 
loss_fn = nn.MSELoss(reduction='mean')

# creat the train_step function for our model , loss function and optimizer 
train_step = make_train_step(model, loss_fn, optimizer)

# create the val_step function for our model and loss function 
val_step = make_val_step(model, loss_fn)
