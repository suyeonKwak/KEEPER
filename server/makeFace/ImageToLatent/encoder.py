# latent to image
import torch
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy as np

from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ImageToLatent(torch.nn.Module):
    def __init__(self, image_size=256):
        super().__init__()

        self.image_size = image_size
        self.activation = torch.nn.ELU()

        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = torch.nn.Sequential(*self.resnet)
        self.conv2d = torch.nn.Conv2d(2048, 256, kernel_size=1)
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(16384, 256)
        # self.dense2 = torch.nn.Linear(256, (18 * 512))
        self.dense2 = torch.nn.Linear(256, 512)

    def forward(self, image):
        x = self.resnet(image)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        # x = x.view((-1, 18, 512))
        x = x.view((-1, 512))

        return x


class ImageLatentDataset(torch.utils.data.Dataset):
    def __init__(self, images, dlatents, image_size=256, transforms=None):
        self.images = images
        self.dlatents = dlatents
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype("float32")
        dlatent = self.dlatents[index]

        image = torch.tensor(image).permute(2, 0, 1)
        dlatent = torch.tensor(dlatent)

        return image, dlatent


def get_encoder(train=False):
    model_path = "./server/model/image_to_latent9.pt"
    image_size = 256

    encoder = ImageToLatent(image_size).to(device)
    encoder.load_state_dict(torch.load(model_path, map_location=device))

    if train:
        encoder.train()
    else:
        encoder.eval()

    return encoder


def get_latent(target_image):
    encoder = get_encoder(train=False)

    tf = T.ToTensor()
    image = tf(target_image).to(device)
    image = torch.clamp(image * 255, 0, 255)

    pred_latent = encoder(image.unsqueeze(0))

    return pred_latent


if __name__ == "__main__":
    model_path = "./server/model/image_to_latent9.pt"
    image_size = 256

    encoder_9 = ImageToLatent(image_size).to(device)
    encoder_9.load_state_dict(torch.load(model_path, map_location=device))
    encoder_9.eval()

    image = Image.open("./server/makeFace/data/Target.png").convert("RGB")
    # image = Image.open('/content/test2.jpg').convert('RGB').resize((256,256))
    tf = T.ToTensor()
    image = tf(image).to(device)
    image = torch.clamp(image * 255, 0, 255)

    # pred_latent = encoder(image.unsqueeze(0))
    pred_latent = encoder_9(image.unsqueeze(0))
    pred_latent = pred_latent.detach().numpy()
    np.save("./server/makeFace/data/target_latent.npy", pred_latent)
