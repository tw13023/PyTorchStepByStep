
# Define number of epochs 
n_epochs = 1000 
losses = []
# For each epoch ...
for epoch in range(n_epochs):
    # inner loop, for each mini batch 
    mini_batch_losses = []
    for x_batch, y_batch in train_loader: 
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # perform one train step and return the corresponding loss
        mini_batch_loss = train_step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    # compute the average loss of the mini batches
    # this is the average loss of the epoch
    loss = np.mean(mini_batch_losses)
    losses.append(loss)
