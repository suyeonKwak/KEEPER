import torch
import torch.nn as nn
import torchvision.transforms as transforms


from collections import OrderedDict

import numpy as np
from PIL import Image
import random

from makeFace.network_modules import G_mapping, G_synthesis

from absl import logging
import os
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.set_verbosity(logging.ERROR)  # warning message 띄우지 않게
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_generator():
    # print('get_generator')
    g_all = nn.Sequential(
        OrderedDict([("g_mapping", G_mapping()), ("g_synthesis", G_synthesis())])
    )

    # load pre-trained weight
    model_path = "./server/model/karras2019stylegan-ffhq-1024x1024.for_g_all.pt"
    g_all.load_state_dict(torch.load(model_path))

    # GPU setting
    g_all.eval()
    g_all.to(device)
    # print('got it!')
    return g_all


def generate_korean_faces(generator, target_latent=None, nb_rows=3, nb_cols=3):
    # print('generate_korean_faces')
    nb_samples = nb_rows * nb_cols

    k_latents = np.load("./server/makeFace/data/k_latent.npy")
    idxs = random.sample(range(1, len(k_latents)), 9)

    latents = torch.from_numpy(k_latents[idxs]).to(device)

    if target_latent != None:
        target_latent = target_latent.repeat(nb_samples, 1)
        latents = latents * 0.5 + target_latent * 0.5

    # Show sample
    with torch.no_grad():
        imgs = generator(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0  # normalization to 0~1 range
    imgs = imgs.cpu()

    return imgs


def generate_faces(generator, target_latent=None, nb_rows=3, nb_cols=3):
    # input setting
    nb_samples = nb_rows * nb_cols

    latents = torch.randn(nb_samples, 512, device=device)
    if target_latent != None:
        target_latent = target_latent.repeat(nb_samples, 1)
        latents = latents * 0.3 + target_latent * 0.7

    # Show sample
    with torch.no_grad():
        imgs = generator(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0  # normalization to 0~1 range
    imgs = imgs.cpu()

    return imgs


def get_latent_vector(generator, image, iteration=100, latent_size=512, lr=0.01):
    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(
                (1024, 1024)
            ),  # Resize the image to match the model's input size
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize the image
        ]
    )
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    # Initialize latent vector randomly
    latent_vector = torch.randn(1, latent_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent_vector], lr=lr)

    # print("Image to latent vector")
    for i in tqdm(range(latent_size)):
        optimizer.zero_grad()

        # 동양인 latent_vector를 찾아 넣어야겠다.
        # 성별 옵션은 여기서 가능할듯
        generated_image = generator(latent_vector)

        # Calculate loss as the L2 distance between generated and target image
        loss = torch.mean((generated_image - image) ** 2)
        loss.backward()
        optimizer.step()

    return latent_vector.detach().cpu().numpy()


def integrated_generated_faces(target_latent, korean=True):
    # print('integrated_generated_faces')
    generator = get_generator()
    # target_latent = get_latent_vector(generator, target_img)
    if korean:
        # print('Yes korean')
        imgs = generate_korean_faces(generator, target_latent=target_latent)
    else:
        imgs = generate_faces(generator, target_latent=target_latent)

    for i, img in enumerate(imgs):
        img = torch.clamp(img * 255, 0, 255)
        img = img.permute(1, 2, 0).detach().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save("./server/makeFace/generated/v{}.png".format(i))


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


if __name__ == "__main__":
    # target 추출 후 업로드
    # video_path = "C:/Users/user/Desktop/Coders/makeFace/data/reference_me.mp4"
    # target_path = "C:/Users/user/Desktop/Coders/makeFace/data/target.png"

    # cam = cv2.VideoCapture(video_path)
    # faseCasecade = cv2.CascadeClassifier(
    #     "C:/Users/user/Desktop/Coders/makeFace/model/haarcascade_frontalface_alt.xml"
    # )  # 여기 바꾸기

    # count = 0

    # while True:
    #     ret, img = cam.read()
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     faces = faseCasecade.detectMultiScale(
    #         gray, scaleFactor=1.2, minNeighbors=9, minSize=(20, 20)
    #     )
    #     if len(faces) == 0:
    #         print("cannot find any face.")
    #     for x, y, w, h in faces:
    #         l = max(w, h)
    #         p = int(l * 0.3)
    #         count += 1
    #         img = img[y - p : y + l + p, x - p : x + l + p]
    #         img = cv2.resize(img, (256, 256))
    #         cv2.imwrite(target_path, img)

    #     print("Complete!")
    #     if count >= 1:
    #         break

    generator = get_generator()
    # target_latent = get_latent_vector(generator, target_img)
    target_latent = np.load(
        "/content/drive/MyDrive/coders/server_yoojin/makeFace/data/target_latent.npy"
    )

    target_latent = torch.Tensor(target_latent)
    imgs = generate_korean_faces(generator, target_latent=target_latent)

    for i, img in enumerate(imgs):
        img = torch.clamp(img * 255, 0, 255)
        img = img.permute(1, 2, 0).detach().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save("./server/makeFace/generated/v{}.png".format(i))

        # file_name = "./generated/v{}.png".format(i)

        # aws_file_name = "Virtual/v{}.png".format(i)

        # s3_upload(file_name, bucket, aws_file_name)
