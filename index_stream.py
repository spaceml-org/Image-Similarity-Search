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
from indexer_class import get_model
global index 

st.title('Image Similarity Search')
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# def increment_counter(model):
#     st.write('INSIDE INCREMENT', model)
#     # model = get_model(model)
#     st.session_state.model = model

if 'indexer' not in st.session_state:
    st.session_state['indexer'] = None
    st.session_state.model = None
    st.session_state['index_file'] = None
with st.form('sidebar'):
    with st.sidebar:
        model = st.file_uploader("Upload your model", type=['pt','pth'])
        st.write('AFTER UPLOAD', model)
        DATA_PATH = st.text_input("Enter data path (absolute)", value= '/Users/tarun/Documents/UCMerced_LandUse/Images')
        embedding_size = int(st.number_input('Model Embedding size', value= 21))
        n_neighbors = int(st.number_input('Num Neighbors', value= 5))
        submitted = st.form_submit_button('Create Index')
        st.write(submitted)
        
if submitted:
        st.write('Creating index file')
        index = Indexer(DATA_PATH, model, embedding_size = embedding_size, device = device)
        index_file = index.get_index()
        st.session_state['indexer'] = index
        st.session_state.model = model
        st.session_state['index_file'] = index_file 
        st.write('Class Initialized ☑️')
        time.sleep(1)

image = st.file_uploader("Upload Query Image", type=['png','jpeg', 'jpg'])
image_upload = st.button('Submit')

if image_upload:
    if image is not None:
        image = read_image(image)
        fig = st.session_state['indexer'].process_image(image)
        st.pyplot(fig=fig, clear_figure=True)

        

    


        




