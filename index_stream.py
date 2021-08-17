from os import read
import streamlit as st
import pandas as pd
import numpy as np
from indexer_class import read_image
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch
import logging
import pickle
import os
import numpy as np
import time
from tqdm.notebook import tqdm
import PIL.Image as Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import faiss
from indexer_class import Indexer  

st.title('Image Similarity Search')
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

with st.form('sidebar'):
    with st.sidebar:
        model = st.file_uploader("Upload your model", type=['pt','pth'])
        DATA_PATH = st.text_input("Enter data path (absolute)")
        embedding_size = int(st.number_input('Model Embedding size'))        
        submitted = st.form_submit_button('Create Index')
        st.write(submitted)

if submitted:
    st.write('Creating index file')
    index = Indexer(DATA_PATH, model, embedding_size = embedding_size, device = device)
    st.write('Class Initialized ☑️')
    time.sleep(1)
    image = st.file_uploader("Upload Query Image", type=['png','jpeg', 'jpg'])
    if image is not None:
        image = read_image(image)
        # st.image(image)
        time.sleep(1)
        fig = index.process_image(image)
        # streamlit.pyplot(fig=fig, clear_figure=False)

        

    


        




