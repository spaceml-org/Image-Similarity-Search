import pickle
import torchvision.datasets as datasets
import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.image as mpimg
from imutils import paths

def load_dependencies(annoy_path,num_nodes):
  u = AnnoyIndex(num_nodes, 'euclidean')
  u.load(annoy_path)

  # with open(embedding_path, 'rb') as handle:
  #   dataset_paths = pickle.load(handle)
  return u

def load_model(ckpt_path, device):

  model = torch.load(ckpt_path)
  model.to('cuda')
  model.eval()
  return model

def inference(model, image_path):
  
  im = Image.open(image_path).convert('RGB')
  image = np.transpose(im, (2,0,1)).copy()
  im = torch.tensor(image).unsqueeze(0).float().cuda()
  x = model(im)

  return x[0][0]

def classname(str):
    return str.split('/')[-2]

def classname_filename(str):
    return str.split('/')[-2] + '/' + str.split('/')[-1]

def plot_images(filenames, distances):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(10, 10))
    columns = 2
    print("Number retrieved",len(images))
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + classname_filename(filenames[i]))
        else:
            ax.set_title("Similar Image\n" + classname_filename(filenames[i]) +
                         "\nDistance: " + str(float("{0:.2f}".format(distances[i]))))
        plt.imshow(image)
        plt.savefig('Nearest_images_' + filenames[0].split('/')[-1]  + '.png')
    plt.show()


def get_nn_annoy(u, image_embedding, features_list_y, n, disp = False):
  
  inds, dists = u.get_nns_by_vector(image_embedding, n ,include_distances = True)
  
  if disp == True:
    for i in range(len(inds)):
      print("Class:",features_list_y[inds[i]].split("/")[-2])
      print("Distance:",dists[i])

  return inds, dists

def main():
  parser = ArgumentParser()
  # parser.add_argument("--image_size", default = 256, type=int, help="Size of the image")
  parser.add_argument("--image_path",type = str, help="Path to image to perform inference")
  parser.add_argument("--ckpt_path",type = str, help="Location of model checkpoint")
  parser.add_argument("--annoy_path",type = str,help="Location of annoy file")
  parser.add_argument("--dataset_pkl_path",type = str, default = None, help="Location of embeddings pickle file")
  parser.add_argument("--device", default = 'cuda', type= str, help="device to run inference on" )
  parser.add_argument("--n_closest", type= int, default = 5, help = "number of closest points")
  parser.add_argument('--data_path', type= str, help = 'Path to dataset')
  args = parser.parse_args()

  image_path = args.image_path
  # size = args.image_size
  ckpt_path = args.ckpt_path
  annoy_path = args.annoy_path
  device = args.device
  n = args.n_closest
  data_path = args.data_path

  model = load_model(ckpt_path,device)
  u = load_dependencies(annoy_path, 512)
  embedding = inference(model, image_path)

  dataset_paths = list(paths.list_images(data_path))
  inds, dists = get_nn_annoy(u, embedding, dataset_paths, n, True)
  chosen_files= [dataset_paths[i] for i in inds]
  chosen_files.insert(0, image_path)
  dists.insert(0, 0.0)
  print('Plotting Images')
  plot_images(chosen_files , dists)


if __name__ == "__main__":
  main()
  
