# latent to image
import torch
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from losses import LogCoshLoss
from makeFace.dataset import make_dataset
from encoder import ImageToLatent, ImageLatentDataset
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(epochs=100):
    train_losses = []
    valid_losses = []
    progress_bar = tqdm_notebook(range(epochs))

    validation_loss = 0.0

    for epoch in progress_bar:
        running_loss = 0.0

        encoder.train()
        for i, (images, latents) in enumerate(train_generator, 1):
            optimizer.zero_grad()
            images, latents = images.cuda(), latents.cuda()

            pred_latents = encoder(images)
            loss = criterion(pred_latents, latents)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(
                "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(
                    i, running_loss / i, validation_loss
                )
            )

            if i == (idx // 32):
                train_losses.append(running_loss / i)

        validation_loss = 0.0

        encoder.eval()
        for i, (images, latents) in enumerate(validation_generator, 1):
            with torch.no_grad():
                images, latents = images.cuda(), latents.cuda()
                pred_latents = encoder(images)
                loss = criterion(pred_latents, latents)

                validation_loss += loss.item()

        validation_loss /= i
        progress_bar.set_description(
            "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(
                i, running_loss / i, validation_loss / i
            )
        )
        valid_losses.append(validation_loss / i)

    return train_losses, valid_losses


def visualize_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 15))
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(train_losses, "b", label="train loss")
    ax[0].set_title("Train Loss")
    ax[0].set_xlabel("steps")
    ax[0].set_ylabel("loss")
    ax[0].set_ylim(0.0, 1.0)

    ax[1].plot(valid_losses, "g", label="val loss")
    ax[1].set_title("Validation Loss")
    ax[1].set_xlabel("steps")
    ax[1].set_ylabel("loss")
    ax[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    samples = 4000
    latents = np.load("./server/makeFace/data/latents_5000.npy")
    images = np.load("./server/makeFace/data/images_5000.npy")
    # latents, imamges = make_dataset(samples)

    # load encoder
    image_size = 256
    encoder = ImageToLatent(image_size).to(device)
    optimizer = torch.optim.Adam(encoder.parameters())
    criterion = LogCoshLoss()

    idx = int(images.shape[0] * 0.8)

    train_images = images[:idx]
    validation_images = images[idx:]

    train_dlatents = latents[:idx]
    validation_dlatents = latents[idx:]

    train_dataset = ImageLatentDataset(train_images, train_dlatents)
    validation_dataset = ImageLatentDataset(validation_images, validation_dlatents)

    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    validation_generator = torch.utils.data.DataLoader(
        validation_dataset, batch_size=32
    )

    train_losses, valid_losses = train(1000)

    visualize_losses(train_losses, valid_losses)

    model_path = "./server/model/image_to_latent9.pt"
    torch.save(encoder.state_dict(), model_path)
