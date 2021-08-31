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
import random
from indexer_class import get_model
global index 

st.title('Image Similarity Search') 
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

image_upload = False
index_progress = False
image = None
# key = str(random.randint(0, 1000000))

def create_temp_file(query, output_file="/tmp/query.png"):
    """create_temp_file.
    :param query:
    :param output_file:
    """
    data = query.read()

    with open(output_file, "wb") as file:
        file.write(data)

    return output_file

def upload_and_process(key):
        display = False
        placeholder = st.empty()
        if st.session_state['index_progress']:
            st.write('key', key)
            image = placeholder.file_uploader("Upload Query Image", type=['png','jpeg', 'jpg'], key = key)
            st.write('func_image', image)
            if image is not None:
                placeholder.empty()
                image = read_image(image)
                st.write(image)
                fig = st.session_state['indexer'].process_image(image, n_neighbors)
                st.pyplot(fig=fig, clear_figure=False)
                display = True
                st.session_state['first_run'] = False
        return display

if 'indexer' not in st.session_state:
    st.session_state['indexer'] = None
    st.session_state.model = None
    st.session_state['index_progress'] = False
    st.session_state['first_run'] = True

with st.form('sidebar'):
    with st.sidebar:
        model = st.file_uploader("Upload your model", type=['pt','pth'])
        DATA_PATH = st.text_input("Enter data path (absolute)", value= '/Users/user/Documents/UCMerced_LandUse/Images')
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
        image = st.file_uploader("Upload Query Image", type=['png','jpeg', 'jpg'])
        submitted_1 = st.form_submit_button('Search')

    if submitted_1:
        uploaded_image = create_temp_file(image)
        image = read_image(image)
        # placeholder.empty()
        fig = st.session_state['indexer'].process_image(image, n_neighbors)
        st.pyplot(fig=fig, clear_figure=False, transparent = True)
        
    # if (label="Search"):
    #     if not image:
    #         st.markdown("Please enter a query")   
    #     else:
            



# st.write('First run', st.session_state['first_run'])
# placeholder = st.empty()
# if st.session_state['first_run']:
#     displayed = upload_and_process('0')

# if not st.session_state['first_run']:
#     placeholder.empty()
#     key = str(random.randint(1, 1000))
#     upload_and_process(key)


        

    

