from os import read
import streamlit as st
import pandas as pd
import numpy as np
from indexer import read_image
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
from indexer import Indexer  
import random
from indexer import get_model
global index 

st.title('Image Similarity Search') 
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def create_temp_file(query, output_file="/tmp/query.png"):
    """create_temp_file.
    :param query:
    :param output_file:
    """
    data = query.read()

    with open(output_file, "wb") as file:
        file.write(data)

    return output_file

if 'indexer' not in st.session_state:
    st.session_state['indexer'] = None
    st.session_state.model = None
    st.session_state['index_progress'] = False
    st.session_state['first_run'] = True

with st.form('sidebar'):
    with st.sidebar:
        model = st.file_uploader("Upload your model", type=['pt','pth'])
        DATA_PATH = st.text_input("Enter data path (absolute)")
        embedding_size = int(st.number_input('Model Embedding size', value= 21))
        n_neighbors = int(st.number_input('Num Neighbors', value= 5))
        submitted = st.form_submit_button('Create Index')
        
if submitted:
        st.write('Creating index file')
        index = Indexer(DATA_PATH, model, embedding_size = embedding_size, device = device)
        st.session_state['indexer'] = index
        st.session_state.model = model
        st.success('Class Initialized ☑️')
        time.sleep(1)
        st.session_state['index_progress'] = True

if st.session_state['index_progress']:
    with st.form('image_similarity'):
        image = st.file_uploader("Upload Query Image", type=['png','jpeg', 'jpg', 'tif'])
        submitted_1 = st.form_submit_button('Search')

    if submitted_1:
        uploaded_image = create_temp_file(image)
        image = read_image(image)
        st.image(image)
        # placeholder.empty()
        fig = st.session_state['indexer'].process_image(image, n_neighbors)
        st.pyplot(fig=fig, clear_figure=False, transparent = True)
        


    

