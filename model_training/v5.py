
# define number of epochs
n_epochs = 200 
lossed = []
val_losses = []

#inner loop, for each mini-batch
for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)

# validation loop - no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)

# record both losses for each epoch under the tag "loss"
    writer.add_scalars(main_tag='loss',
                        tag_scalar_dict= {
                            'training': loss,
                            'validation': val_loss
                        },
                        global_step = epoch)

writer.close()
