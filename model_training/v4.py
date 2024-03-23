# define number of epochs 
n_epochs = 200 

losses = []
val_losses = []

# outer loop , for each epoch 
for epoch in range(n_epochs):
    # inner loop. for mini batches 
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)
    # validation loop - no gradients in validation !
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)
