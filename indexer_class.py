from torch.utils.data import DataLoader
from torchvision import transforms
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

if torch.cuda.is_available():
    device = 'cuda'
else :
    device = 'cpu'

class Indexer:
    def __init__(self, img, DATA_PATH, model_path, img_size = 224, batch_size = 128, 
                embedding_size = 128, device = device) -> None:
        
        self.model = get_model(model_path)
        self.DATA_PATH = DATA_PATH
        
        self.embeddings, images_list = get_matrix(self.model, self.DATA_PATH, img_size, embedding_size)
        self.images_list = [path[0] for path in images_list]
        self.index = index_gen()
          # TODO Perform caching till this step, as an init and automatically process packets as they come in. Convert to a class when integrating into streamlit 
        
    def process_image(self, image_path, n_neighbors = 5):
        src = get_embedding(self.model, image_path)
        scores, neighbours = self.index.search(x=src,k= n_neighbors)
        fig = get_fig(neighbours[0],self.images_list)
        #TODO do a streamlit write for this returned subplot


def get_model(model_path) -> dict:
    model = torch.load(model_path) if device == 'cuda' else torch.load(model_path, map_location = 'cpu')
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
    size = embedding_size
    with torch.no_grad():
        data_matrix = torch.empty(size = (0, size)).cuda() if device == 'cuda' else torch.empty(size = (0, size))
        bs = 128
        if len(dataset) < bs:
          bs = 1
        loader = DataLoader(dataset, batch_size = bs, shuffle = False)
        for batch in tqdm(loader):
            x = batch[0].cuda() if device == 'cuda' else batch[0]
            embeddings = model(x)
            data_matrix = torch.cat([data_matrix, embeddings])
    return data_matrix.cpu().detach().numpy(), dataset.imgs

def index_gen(embeddings):
  d = 512
  nb = 1363016
  index = faiss.IndexFlatL2(d)
  index.add(embeddings)
  faiss.write_index(index,'index.bin')
  logging.info("Index created. Stored as index.bin")
  return index 


def get_fig(neighbours,images_list):
  fig, axarr = plt.subplots(len(neighbours),1,figsize=(15,30))
  plt.axis('off')
  for i in range(len(neighbours)):
    im = Image.open(images_list[neighbours[i]]).convert('RGB')
    # axarr[i].imshow(im)
    axarr[i].axis('off')
    axarr[i].set_title(neighbours[i])
  return fig #Returns fig to directly push to streamlit

def get_embedding(model,image):
  def to_tensor(pil):
      return torch.tensor(np.array(pil)).permute(2,0,1).float()

  t = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.Lambda(to_tensor)
                          ])
  model.eval()
  if device == 'cuda':
        model.cuda()
  im = Image.open(image).convert('RGB')
  datapoint = t(im).unsqueeze(0).cuda() if device == 'cuda' else t(im).unsqueeze(0) #only a single datapoint so we unsqueeze to add a dimension
  with torch.no_grad():
    print(datapoint)
    embedding = model(datapoint) #get embedding
  return embedding.detach().cpu().numpy()