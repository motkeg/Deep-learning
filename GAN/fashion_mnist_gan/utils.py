import numpy as np
import matplotlib.pyplot as plt

 # Plot the loss from each batch
def plot_loss(epoch,d_Losses,g_Losses):
    plt.figure(figsize=(10, 8))
    plt.plot(d_Losses, label='Discriminitive loss')
    plt.plot(d_Losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated images
def plot_generated_images(imgs , epoch, examples=25, dim=(5,5), figsize=(5, 5)):

    plt.figure(figsize=figsize)
    for i in range(imgs.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(imgs[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./samples/fashion_mnist/dcgan_generated_image_epoch_%d.png' % epoch)
