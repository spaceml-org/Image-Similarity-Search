<a href="https://ibb.co/gjntjtx"><img src="https://i.ibb.co/98My8y7/Screenshot-2021-09-01-at-3-16-59-PM.png" alt="Screenshot-2021-09-01-at-3-16-59-PM" border="0"  width="100%"></a>

Image Similarity Search is an app that helps perform super fast image retrieval on PyTorch models for better embedding space interpretability.


## How it works
<a href="https://ibb.co/jrmg7Bw"><img src="https://i.ibb.co/0GThP89/Screenshot-2021-09-01-at-7-27-49-PM.png" alt="Screenshot-2021-09-01-at-7-27-49-PM" border="0"></a>

That's it really. There are functions in the two files provided that: 
* Generate embeddings from your model based on your Dataset.
* Index your Image embeddings to an FAISS index file.  
* Performs Similarity Search to retrieve n closest images to your query.
* Visually shows the images chosen, and allows to search with the index file several times.

## Usage

The app was built with streamlit, and it can be run locally by launching a Streamlit server from the repository directory. 

```
streamlit run app.py
```
### Steps

1. Upload the model file in .pt or .pth format. (Ignore default file limit)
Note : State dicts are not supported due to the underlying class dependency. 
2. Enter absolute path of dataset to be indexed. Note : Dataset must be in PyTorch [`ImageFolder`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder) format. 
3. Enter the output embedding size of the model. Eg. Global Average Pooling layer of a ResNet outputs 2048 dim vectors. 
4. Enter the number of neighbours to be displayed for the given query image. 
5. Please wait while the index is generated. It is stored as `index.bin` in your working directory. 
6. Upload the query image and view similar images from your dataset as indexed by your model. 

## Samples

The repository contains a sample model trained on the UC Merced LandUse Dataset for quick demonstration of Image Similarity Search. Use the model under `samples/uc_merced.pt`
and download the dataset using the command 

```
wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
unzip -qq UCMerced_LandUse.zip
```


## Dependencies

Install the necessary packages from requirements.txt using ```pip install -r requirements.txt``` before you run the scripts.

The app is supported on both CPU and CUDA enabled devices. 
## TODO
- [X] Add FAISS Support
- [X] Command Line Tool -> Streamlit App
- [ ] Allow uploading existing indices
- [ ] Enable Interactive TSNE plots 


*Pull requests are more than welcomed!*


