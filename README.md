<a href="https://ibb.co/Qb7NhCk"><img src="https://i.ibb.co/f1VpZn9/Image-Similarity-Search.png" alt="Image-Similarity-Search" border="0" width="100%"></a>
Image Similarity Search is a tool that performs super fast image retrieval on PyTorch models for better embedding space interpretability.



## How it works
<a href="https://ibb.co/17Wpdpz"><img src="https://i.ibb.co/CmGXMXP/how-it-works.png" alt="how-it-works" border="0"></a>

That's it really. There are functions in the two files provided that: 
* Generate embeddings from your model based on your Dataset.
* Index your Image embeddings to an Annoy tree. 
* Performs Similarity Search on the Annoy tree to retrieve n closest images to your query.
* Visually shows them on a plot that is saved in the current working directory.

## Usage

There are two scripts we've to run to perform the similarity search:

### 1. Indexer
Simply run the ```indexer.py``` to index your data with the following arguments:
* --data_path : Path to Dataset in ```ImageFolder``` structure
* --ckpt_path : Path to PyTorch model
* --annoy_path : Path where the Annoy Tree gets stored with filename
* --embedding_size : Output embedding size of your model 
* --device : cuda/cpu 

and a few other optional arguments (check file).

***You need to run this only one time to create the annoy index.*** 



Example: 
```bash
python indexer.py --DATA_PATH "/content/UCMerced_LandUse/Images"   --embedding_size 512 --ckpt_path "/content/pytorch_model.pt" --annoy_path "/content/annoy_file.ann" 
```

### 2. Retrieval 
Simply run the ```search_by_example.py``` to perform Image Retrieval on a query image with the following arguments:
* --image_path : Query Image path
* --data_path : Path to Dataset in ```ImageFolder``` structure
* --ckpt_path : Path to PyTorch model (**Not a StateDict**)
* --annoy_path : Path where the Annoy Tree was stored after running ```indexer.py```
* --n_closest : Retrieves n closest images from the query image

and a few other optional arguments (check file)

Running this script will generate a matplotlib visualisation containing the query and the retrieved images. Check out [_this example_](https://i.ibb.co/TPbbxf5/image.png). 


Example: 
```bash
python search_by_example.py --image_path '/content/Images/forest03.tif' --ckpt_path '/content/pytorch_model.pt' --annoy_path "/content/annoy_file.ann" --data_path "/content/UCMerced_LandUse/Images" --n_closest 5  
```



## TODO
- [X] Add Annoy Support
- [ ] Add TensorFlow Support    
- [ ] Web Client 
- [ ] Enable Interactive TSNE plots 


*Pull requests are more than welcomed!*


