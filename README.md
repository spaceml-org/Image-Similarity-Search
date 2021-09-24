<a href="https://ibb.co/gjntjtx"><img src="https://i.ibb.co/98My8y7/Screenshot-2021-09-01-at-3-16-59-PM.png" alt="Screenshot-2021-09-01-at-3-16-59-PM" border="0"  width="100%"></a>

---
<div align="center">

<p align="center">
  Published by <a href="http://spaceml.org/">SpaceML</a> •
  <a href="https://arxiv.org/abs/2012.10610">About SpaceML</a>
</p>

[![Python Version](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7%20|%203.8-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/Cuda-10%20|%2011.0-4dc71f.svg)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
[![Pip Package](https://img.shields.io/badge/Pip%20Package-Coming%20Soon-0073b7.svg)](https://pypi.org/project/pip/)
[![Docker](https://img.shields.io/badge/Docker%20Image-Coming%20Soon-34a0ef.svg)](https://www.docker.com/)
</div>

# What is Image Similarity Search?

Similarity search operations involve applying operations such as Nearest Neighbour search on latent embeddings of a dataset. 

**Image Similarity Search** is an app that helps perform super fast image retrieval on PyTorch models for better embedding space interpretability. It takes a pre-trained PyTorch model, a dataset, and a query image, and retrives similar examples within the dataset for the given query image, using the pretrained model.


# How does it work?
<a href="https://ibb.co/MNT50qW"><img src="https://i.ibb.co/tCfPryk/Screenshot-2021-09-01-at-8-23-35-PM.png" alt="Screenshot-2021-09-01-at-8-23-35-PM" border="0"></a>

That's it really. There are functions in the two files provided that: 
1. Generate embeddings from your model based on your Dataset.
2. Index your Image embeddings to an FAISS index file.  
3. Performs Similarity Search to retrieve N closest images to your query.
4. Visually shows the nearest images, and the app allows you to search with the index file several times.

# Usage

The app was built with [Streamlit](streamlit.io), and it can be run locally by launching a Streamlit server from the repository directory. If you do not have Streamlit installed, follow the steps under the **Dependencies** section before you get started. 

## Dependencies

Install the necessary packages from requirements.txt using 
```
pip install -r requirements.txt
```  


The app uses Facebook AI's FAISS package to perform similarity search. Install that using the instructions given [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) based on your hardware. 

The app is supported on both CPU and CUDA enabled devices. 


<a href="https://ibb.co/Ct0Wxrz"><img src="https://i.ibb.co/PYCtBSN/Screenshot-2021-09-24-at-12-43-18-PM.png" alt="Screenshot-2021-09-24-at-12-43-18-PM" border="0" width="100%"></a>

```
streamlit run app.py
```
## Steps

(Ref. image above)
1. Upload the model file in .pt or .pth format. (Ignore default file limit)
Note : State dicts are not supported due to the underlying class dependency. 
2. Enter absolute path of dataset to be indexed. Note : Dataset must be in PyTorch [`ImageFolder`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder) format. 
3. Enter the output embedding size of the model. Eg. Global Average Pooling layer of a ResNet outputs 2048 dim vectors. 
4. Enter the number of neighbours to be displayed for the given query image. 
5. Please wait while the index is generated. It is stored as `index.bin` in your working directory. 
6. Upload the query image and view similar images from your dataset as indexed by your model. 

## [Samples](samples/sample.md)

The repository contains a sample model trained on the UC Merced LandUse Dataset for quick demonstration of Image Similarity Search. Use the model under `samples/uc_merced.pt` and download the dataset using the command 

```
wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
unzip -qq UCMerced_LandUse.zip
```
The dataset is downloaded and unzipped in your present working directory.



# TODO
- [X] Add FAISS Support
- [X] Command Line Tool -> Streamlit App
- [ ] Allow uploading existing indices
- [ ] Enable Interactive TSNE plots 


*Pull requests are more than welcomed!*


