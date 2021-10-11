# Samples

The repository contains a sample model trained on the UC Merced LandUse Dataset for quick demonstration of Image Similarity Search. Use the model under `samples/uc_merced.pt`
and download the dataset using the command 

```
wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
unzip -qq UCMerced_LandUse.zip
```
Default embedding size is 21. 

NOTES : 

* While creating your model, use the output embedding size from the forward() function. 
* State dicts are not directly supported in the current version, but feel free to change the code under `indexer.py` to load the model the way you want. 