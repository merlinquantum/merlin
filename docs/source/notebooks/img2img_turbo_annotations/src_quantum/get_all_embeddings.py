import os

import h5py
import torch
import torchvision.transforms.functional as F
from cyclegan_turbo import (
    CycleGAN_Turbo,
)
from my_utils.training_utils import (
    build_transform,
)
from PIL import Image

# BosonSampler
from quantum_encoder import BosonSampler, QEncoder
from tqdm.auto import tqdm

# small functions ###


# load images and paths from Dataset
def load_images(dataset, idx):
    weight_dtype = torch.float32
    img_A = dataset[idx]["pixel_values_src"].to(dtype=weight_dtype).to("cuda:0")
    img_A_path = dataset[idx]["path_src"]
    img_B = dataset[idx]["pixel_values_tgt"].to(dtype=weight_dtype).to("cuda:0")
    img_B_path = dataset[idx]["path_tgt"]
    img_A = img_A.unsqueeze(0)
    img_B = img_B.unsqueeze(0)
    return img_A, img_B, img_A_path, img_B_path


# forward one img through the encoder
def forward_image(img, model, direction):
    output_enc = model.vae_enc(img, direction=direction)
    # apply sigmoid for entry of boson sampler
    output_enc = torch.sigmoid(output_enc)
    output_enc = output_enc.squeeze(0)
    return output_enc


# forward the classical embeddings through the quantum encoder
def get_quantum_embeddings(emb, bs, qencoder):
    quantum_embs = bs.compute(qencoder.encode(emb))
    decoded = qencoder.decode(quantum_embs)
    # apply logit to transfer to correct domain
    decoded = torch.logit(decoded)
    return decoded


def prepare_image(img_path, transform):
    img_pil = Image.open(img_path).convert("RGB")
    img_t = F.to_tensor(transform(img_pil))
    img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
    return img_t


# GENERAL FUNCTION ###


def get_dataset_embeddings(model_path, target_folder):
    # create target folder to store the embeddings
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # load classical model ###
    # sd = torch.load(quantum_start_path)
    cyclegan_q = CycleGAN_Turbo(pretrained_path=model_path).to("cuda:0")
    print("--- Model loaded ---")

    # dataset and image for test ###
    transform = build_transform(image_prep="resize_128")
    data_folder = "../data/dataset_full_scale/"
    train_images_A = os.path.join(data_folder, "test_A")
    # train_images_B = os.path.join(data_folder, "test_B")
    weight_dtype = torch.float32

    # tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False,)
    # dataset_train = UnpairedDataset(dataset_folder="../data/dataset_full_scale/", image_prep="resize_128", split="train", tokenizer=tokenizer)
    print("--- Data loaded ---")

    # quantum encoder ###
    # Set the random seeds
    torch.manual_seed(42)
    dims = (4, 16, 16)
    qencoder = QEncoder(dims)
    bs = BosonSampler(qencoder.m, qencoder.n)
    print("--- Quantum encoder defined ---")

    # len_images = len(dataset_train)
    for img_A_name in tqdm(os.listdir(train_images_A), desc="Processing"):
        # for img_A_name in ["0001542f-5ce3cf52.jpg","0001542f-7c670be8.jpg"]:
        img_A_path = os.path.join(train_images_A, img_A_name)
        img_A = prepare_image(img_A_path, transform)
        img_A = img_A.to(dtype=weight_dtype).to("cuda:0")
        img_A = img_A.unsqueeze(0)
        # get classical embeddings
        cl_embeddings_A = forward_image(img_A, cyclegan_q, "a2b")

        # get quantum embeddings
        qu_embeddings_A = get_quantum_embeddings(cl_embeddings_A, bs, qencoder)

        # write to H5
        with h5py.File(os.path.join(target_folder, "test_A.h5"), "a") as f:
            d_name = os.path.basename(img_A_path)
            if d_name in f:
                del f[d_name]
            _dataset = f.create_dataset(
                d_name, data=qu_embeddings_A, compression="gzip"
            )

    print("-- Quantum embeddings A to H5 completed --")
    # for img_B_name in tqdm(os.listdir(train_images_B),desc = "Processing"):
    #     img_B_path = os.path.join(train_images_B,img_B_name)
    #     img_B = prepare_image(img_B_path,transform)
    #     img_B= img_B.to(dtype=weight_dtype).to("cuda:0")
    #     img_B = img_B.unsqueeze(0)
    #     # get classical embeddings
    #     cl_embeddings_B = forward_image(img_B,cyclegan_q, "b2a")
    #     # get quantum embeddings
    #     qu_embeddings_B = get_quantum_embeddings(cl_embeddings_B, bs, qencoder)

    #     # write to H5
    #     with h5py.File(f"{target_folder_B}.h5","a") as f:
    #         d_name = os.path.basename(img_B_path)
    #         if d_name in f:
    #             del f[d_name]
    #         dataset = f.create_dataset(d_name,data = qu_embeddings_B,compression="gzip" )
    # print("-- Quantum embeddings B to H5 completed --")
    print("-- done writing all quantum embeddings to h5 --")


# path to weights
quantum_start_path = "/home/jupyter-pemeriau/img2img-turbo/all_outputs/exp-109/checkpoints/model_1001.pkl"
# target_folder
# target = "/home/jupyter-pemeriau/q_embs/all_emb_128_dims_4_16_16_ckpt1001"
target = "/home/jupyter-pemeriau/q_embs/all_emb_128_dims_4_16_16_ckpt1001_42"
get_dataset_embeddings(quantum_start_path, target)


# READ THE DATA ###

# with h5py.File("/home/jupyter-pemeriau/q_embs/test4/train_A.h5", "r") as f:
#     # Get a list of all dataset names
#     dataset_names = list(f.keys())
#     print(f"Names = {dataset_names}")

#     # Iterate over the dataset names and load the corresponding tensors
#     for dataset_name in dataset_names:
#         dataset = f[dataset_name]
#         tensor = dataset[:]  # Load the entire dataset as a NumPy array

#         # Do something with the loaded tensor
#         print(f"Dataset name: {dataset_name}, Tensor shape: {tensor.shape}")

# save tensor
# tensor_img = cl_embeddings_A.permute(1, 2, 0)
# numpy_img = tensor_img.cpu().detach().numpy()
# numpy_img = (numpy_img * 255).astype(np.uint8)  # Convert to uint8
# img = Image.fromarray(numpy_img)
# img.save(f'{target_folder_A}/{img_A_name}-emb.png')
