from torch.utils.data import DataLoader
from annoy import AnnoyIndex
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


def get_embeddings_test(DATA_PATH, ckpt, size, embedding_size):
  ims = []
  for folder in os.listdir(DATA_PATH):
    for im in os.listdir(f'{DATA_PATH}/{folder}'):
      ims.append(f'{DATA_PATH}/{folder}/{im}')
  
  model = torch.load(ckpt)
  model.eval()
  model.cuda()
  t = transforms.Resize((size, size))
  embedding_matrix = torch.empty(size= (0, embedding_size)).cuda()
  
  for f in tqdm(ims):
    with torch.no_grad():
      im = Image.open(f).convert('RGB')
      im = t(im)
      im = np.asarray(im).transpose(2, 0, 1)
      im = torch.Tensor(im).unsqueeze(0).cuda()
      embedding = model(im)[0]
      embedding_matrix = torch.vstack((embedding_matrix, embedding))
  print('Embedding Shape', embedding_matrix.shape)
  return embedding_matrix.detach().cpu().numpy()


def prepare_tree(num_nodes, features_list_x, path, num_trees):
  t = AnnoyIndex(num_nodes, 'euclidean')
  for i in range(len(features_list_x)):
    t.add_item(i,features_list_x[i])
  t.build(num_trees)
  t.save(path)
  return path

def pickle_filepaths(dataset_paths,pickle_path):
  with open(pickle_path, 'wb') as handle:
    pickle.dump(dataset_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return pickle_path

def main():

  parser = ArgumentParser()
  parser.add_argument("--image_size", default = 256, type=int, help="Size of the image")
  parser.add_argument("--DATA_PATH",type = str, help="Path to image to perform inference")
  parser.add_argument("--ckpt_path",type = str, help="Location of model checkpoint")
  parser.add_argument("--annoy_path",type = str,help="Location to save annoy file (end with .ann)")
  parser.add_argument("--embedding_size", default= 128, type = int, help="Image size for embedding")
  parser.add_argument("--device", default = 'cuda', type= str, help="device to run inference on" )
  parser.add_argument("--num_nodes",type = int, help="Number of nodes in the final dense layer of the model")
  parser.add_argument("--num_trees",default = 50, type = int, help="Number of trees in Annoy")
  parser.add_argument("--batch_size",default =64, type = int, help="Batch Size for dataloader")

  args = parser.parse_args()
  DATA_PATH = args.DATA_PATH
  size = args.image_size
  ckpt_path = args.ckpt_path
  annoy_path = args.annoy_path
  num_nodes = args.num_nodes
  batch_size = args.batch_size
  emb_size = args.embedding_size
  num_trees = args.num_trees
  device = args.device

  embedding = get_embeddings_test(DATA_PATH, ckpt_path, size, emb_size)
  print("Annoy file stored at",prepare_tree(num_nodes, embedding, annoy_path, num_trees = num_trees))

if __name__ == "__main__":
  main()

