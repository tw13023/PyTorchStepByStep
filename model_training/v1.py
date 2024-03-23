# define number of epochs 
n_epochs = 1000 
losses = []

# For each epoch ....
for epoch in range(n_epochs):
    # perform one train step and retrun the correspoding loss 
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)
