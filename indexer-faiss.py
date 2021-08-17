from indexer import get_embeddings_test
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import ArgumentParser

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

def get_model(model_path):
    model = torch.load(model_path) if device == 'cuda' else torch.load(model_path, map_location = 'cpu')
    return model 

def get_matrix(model, DATA_PATH, image_size = 224, embedding_size = 2048):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()

    
    t = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.Lambda(to_tensor)
                            ])

    dataset = ImageFolder(DATA_PATH, transform = t)
    # model = load_checkpoint(MODEL_PATH)
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

def process_image(model, image_path, index, n_neighbors = 5, images_list):
  src = get_embedding(model, image_path)
  scores, neighbours = index.search(x=src,k= n_neighbors)

  fig = get_fig(neighbours[0],images_list)
  #TODO streamlit write pyplot on screen


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
  model.cuda()
  im = Image.open(image).convert('RGB')
  plt.imshow(im) # TODO Not required during streamlit version
  datapoint = t(im).unsqueeze(0).cuda() #only a single datapoint so we unsqueeze to add a dimension
  with torch.no_grad():
    print(datapoint)
    embedding = model(datapoint) #get embedding
  return embedding.detach().cpu().numpy()


def main():

  parser = ArgumentParser()
  parser.add_argument("--image_path", type=str)
  parser.add_argument("--image_size", default = 224, type=int, help="Size of the image")
  parser.add_argument("--DATA_PATH",type = str, help="Path to image to perform inference")
  parser.add_argument("--ckpt_path",type = str, help="Location of model checkpoint")
  parser.add_argument("--embedding_size", default= 128, type = int, help="Image size for embedding")
  parser.add_argument("--device", default = 'cuda', type= str, help="device to run inference on" )

  args = parser.parse_args()
  DATA_PATH = args.DATA_PATH
  image_size = args.image_size
  model_path = args.ckpt_path
  batch_size = args.batch_size
  emb_size = args.embedding_size
  num_trees = args.num_trees
  device = args.device
  image_path = args.image_path
  
  model = get_model(model_path)

  # embedding = get_matrix(model, DATA_PATH, image_size, emb_size)
  embeddings, images_list = get_matrix(model, DATA_PATH, image_size, emb_size)
  images_list = [path[0] for path in images_list]
  index_gen()



