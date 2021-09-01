from torch.utils.data import DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch
import logging
import pickle
import os
import numpy as np
from tqdm.notebook import tqdm
import PIL.Image as Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import faiss
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Indexer:
    # @st.cache(suppress_st_warning=True)
    def __init__(self, DATA_PATH, model, img_size = 224, embedding_size = 128, device = device) -> None:
        self.model = get_model(model)
        self.DATA_PATH = DATA_PATH
        st.write('Generating embeddings')
        self.embeddings, images_list = get_matrix(self.model, self.DATA_PATH, img_size, embedding_size)
        self.images_list = [path[0] for path in images_list]
        self.index = index_gen(self.embeddings)
          # TODO Perform caching till this step, as an init and automatically process packets as they come in. Convert to a class when integrating into streamlit 

    def process_image(self, img, n_neighbors = 5):
        src = get_embedding(self.model, img)
        scores, neighbours = self.index.search(x=src, k= n_neighbors)
        fig = get_fig(neighbours[0],self.images_list)
        #TODO do a streamlit write for this returned subplot

def get_model(model) -> dict:

    model = torch.load(model) if device == 'cuda' else torch.load(model, map_location = 'cpu')
    return model

def get_matrix(model, DATA_PATH, image_size = 224, embedding_size = 2048) -> np.ndarray:
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()

    t = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.Lambda(to_tensor)
                            ])

    dataset = ImageFolder(DATA_PATH, transform = t)
    model.eval()
    if device == 'cuda':
        model.cuda()
    size = embedding_size #@Ajay TODO Check this please. What should be the embedding size?
    with torch.no_grad():
        data_matrix = torch.empty(size = (0, size)).cuda() if device == 'cuda' else torch.empty(size = (0, size))
        bs = 128
        if len(dataset) < bs:
          bs = 1
        loader = DataLoader(dataset, batch_size = bs, shuffle = False)
        my_bar = st.progress(0)
        for i, batch in enumerate(loader):
            x = batch[0].cuda() if device == 'cuda' else batch[0]
            embeddings = model(x)
            data_matrix = torch.cat([data_matrix, embeddings])
            my_bar.progress(int(100 * i / len(loader))) 
        my_bar.progress(100)
    return data_matrix.cpu().detach().numpy(), dataset.imgs

def index_gen(embeddings):
    d = embeddings.shape[-1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, 'index.bin')
    st.write("Index created. Stored as index.bin")
    return index

def get_fig(neighbours, images_list):
    fig, axarr = plt.subplots(1, len(neighbours), figsize = (7, 7))
    plt.axis('off')
    for i in range(len(neighbours)):
        im = Image.open(images_list[neighbours[i]]).convert('RGB')
        axarr[i].imshow(im)
        axarr[i].axis('off')
    return fig #Returns fig to directly push to streamlit #Returns fig to directly push to streamlit
    

def get_embedding(model,im):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()

    t = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.Lambda(to_tensor)
                            ])
    model.eval()
    if device == 'cuda':
            model.cuda()
    datapoint = t(im).unsqueeze(0).cuda() if device == 'cuda' else t(im).unsqueeze(0) #only a single datapoint so we unsqueeze to add a dimension
    with torch.no_grad():
        embedding = model(datapoint) #get_embedding
    return embedding.detach().cpu().numpy()

def read_image(img_byte):
    img = Image.open(img_byte).convert('RGB')
    return img