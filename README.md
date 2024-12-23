
### tumor Region recognition: Identify cancer area for each WSI

* `camelyon16xml2json.py`: Convert contour annotation from xml format to json format.
* `tissue_mask.py`: Get the tissue area of WSI.
* `tumor_mask.py`: Get the cancer area mask of WSI.
* `non_tumor_mask.py`: Get the normal area mask of WSI.
* `sampled_spot_gen.py`:Get random coordinate points in the cancer region
* `patch_gen.py`:Get the patch dataset
* `train.py`：Train cancerous area segmentation model on cancer and normal patch dataset.
* `probs_map.py`：Predicting the cancerous area heatmap of WSI using cancerous area segmentation model.

### BYOL: Use BYOL for patch embedding

* `train_byol.py`：Training the BYOL model on our pathological iamge (patch) dataset.
* `pre_byol.py`：Use the trained BYOL for patch embedding.
* `pca.py`：feature dimension reduction

### AMIL：AMIL training

* `make_data_list_mil.py`：Make and split training set, validation set and test set for AMIL model.
* `amil_train.py`：Training the AMIL-Patch model.

- - - 
### Environments
* Python==3.8.3
* Ubuntu==18.04.5
* torch==1.7.0
* torchvision==0.8.1
* timm==0.3.2
* imageio==2.25.1
* matplotlib==3.6.2
* numpy==1.23.0
* opencv-python==4.6.0.66
* pandas==1.5.2
* Pillow==9.4.0
* scikit_image==0.17.2
* scikit_learn==1.2.1
* scipy==1.10.1
* skimage==0.0
* spams==2.6.5.4
* tensorboardX==2.6
* pytorch_lightning==1.9.3
* openslide_python==1.1.2
- - -
