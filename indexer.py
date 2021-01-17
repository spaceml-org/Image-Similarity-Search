from torch.utils.data import DataLoader
from annoy import AnnoyIndex
from torchvision import transforms
from argparse import ArgumentParser

import torchvision.datasets as datasets
import torch
import pickle


def build_loader(DATA_PATH, image_size,batch_size):

  transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size], interpolation=2)])
  val_data = datasets.ImageFolder(root=DATA_PATH,transform=transform) 
  dataset_paths = [pair[0] for pair in val_data.samples]
  val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=False, drop_last=False)

  return val_loader, dataset_paths

def get_embeddings(ckpt_path,data_loader, device):

  embedding = []
  simclr_model = torch.load(ckpt_path)
  simclr_model.eval()
  # simclr_model.to(device)
  for step,(x,y) in enumerate(data_loader):
    x = x.to(device)
    # print(simclr_model(x))
    embedding.extend(simclr_model(x)[0].detach().cpu().numpy())
  print("Number of Embeddings:", len(embedding))
  print("Shape of Single Embedding:",embedding[0].shape)
  print(embedding[0])
  return embedding

def prepare_tree(num_nodes, features_list_x, path, num_trees):
  t = AnnoyIndex(num_nodes, 'euclidean')
  for i in range(len(features_list_x)):
    t.add_item(i,features_list_x[i])
  t.build(num_trees)
  t.save(path)
  # print("Tree saved at:",path)
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
  # parser.add_argument("--embedding_path",type = str, help="Location to save embeddings pickle file")
  parser.add_argument("--device", default = 'cuda', type= str, help="device to run inference on" )
  parser.add_argument("--num_nodes",type = int, help="Number of nodes in the final dense layer of the model")
  parser.add_argument("--num_trees",default = 50, type = int, help="Number of trees in Annoy")
  parser.add_argument("--batch_size",default =64, type = int, help="Batch Size for dataloader")

  args = parser.parse_args()
  DATA_PATH = args.DATA_PATH
  size = args.image_size
  ckpt_path = args.ckpt_path
  annoy_path = args.annoy_path
  # embedding_path = args.embedding_path
  num_nodes = args.num_nodes
  batch_size = args.batch_size
  num_trees = args.num_trees
  device = args.device

  data_loader, dataset_paths = build_loader(DATA_PATH, size, batch_size)
  embedding = get_embeddings(ckpt_path, data_loader, device)
  print("Annoy file stored at",prepare_tree(num_nodes, embedding, annoy_path, num_trees = num_trees))
  # print("Embeddings Pickle file stored at",pickle_filepaths(dataset_paths,embedding_path))

if __name__ == "__main__":
  main()

