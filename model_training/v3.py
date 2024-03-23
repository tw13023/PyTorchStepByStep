n_epochs = 200 
losses = [] 
for epoch in range(n_epochs):
    # inner loop, run thru all the mini batches
    loss = mini_batch(device, train_loader , train_step)
    losses.append(loss)
