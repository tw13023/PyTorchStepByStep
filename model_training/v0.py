# model training 
# define number of epochs 
n_epochs = 1000

# set model into train mode 
for epoch in range(n_epochs):

    # Step 1 forward pass, compute the prediction 
    model.train()
    yhat = model(x_train_tensor)
    
    # Step 2 compute the loss 
    loss = loss_fn(yhat, y_train_tensor)

    # Step 3 backward pass, compute the gradients
    loss.backward()

    # Step 4 update the parameters using gradients and the learning rate
    optimizer.step()
    optimizer.zero_grad()
