from indexer import get_embeddings_test
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import ArgumentParser

import torchvision.datasets as datasets
import torch
import pickle
import os
import numpy as np
from tqdm.notebook import tqdm
import PIL.Image as Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

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

def main():

  parser = ArgumentParser()
  parser.add_argument("--image_size", default = 224, type=int, help="Size of the image")
  parser.add_argument("--DATA_PATH",type = str, help="Path to image to perform inference")
  parser.add_argument("--ckpt_path",type = str, help="Location of model checkpoint")
  parser.add_argument("--embedding_size", default= 128, type = int, help="Image size for embedding")
  parser.add_argument("--device", default = 'cuda', type= str, help="device to run inference on" )
  parser.add_argument("--num_trees",default = 50, type = int, help="Number of trees in Annoy")

  args = parser.parse_args()
  DATA_PATH = args.DATA_PATH
  image_size = args.image_size
  model_path = args.ckpt_path
  num_nodes = args.num_nodes
  batch_size = args.batch_size
  emb_size = args.embedding_size
  num_trees = args.num_trees
  device = args.device
  
  model = get_model(model_path)
  

  embedding = get_matrix(model, DATA_PATH, image_size, emb_size)
  print("Annoy file stored at",prepare_tree(num_nodes, embedding, annoy_path, num_trees = num_trees))

def get_fig(neighbours,images_list):
  fig, axarr = plt.subplots(len(neighbours),1,figsize=(15,30))
  plt.axis('off')
  for i in range(len(neighbours)):
    im = Image.open(images_list[neighbours[i]]).convert('RGB')
    # axarr[i].imshow(im)
    axarr[i].axis('off')
    axarr[i].set_title(neighbours[i])

  return fig

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
  plt.imshow(im)
  datapoint = t(im).unsqueeze(0).cuda() #only a single datapoint so we unsqueeze to add a dimension
  with torch.no_grad():
    print(datapoint)
    embedding = model(datapoint) #get embedding
  return embedding.detach().cpu().numpy()