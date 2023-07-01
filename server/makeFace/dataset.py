import torch
from PIL import Image
from main import get_generator
from tqdm import tqdm
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_dataset(num):
    generator = get_generator()

    latents = torch.randn(num, 512, device="cpu")
    imgs = []

    for i, latent in enumerate(tqdm(latents)):
        latent = latent.unsqueeze(0)
        latent = latent.to(device)

        # Show sample
        with torch.no_grad():
            img = generator(latent)
            img = (img.clamp(-1, 1) + 1) / 2.0  # normalization to 0~1 range

        img = img.cpu().squeeze()
        img = torch.clamp(img * 255, 0, 255)
        img = img.permute(1, 2, 0).detach().numpy().astype(np.uint8)
        # print(img.shape)
        img = Image.fromarray(img).resize((256, 256))
        img = np.array(img)
        imgs.append(img)

    # convert as numpy
    latents = latents.numpy()
    imgs = np.array(imgs)

    return latents, imgs
