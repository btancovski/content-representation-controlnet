# Content Representatioin ControlNet

## Creating Content Representations of Images
This section shows the setup for generating a Content Representation of an image. 

### Requirements: 

- Ubuntu 16.04 or higher (tested on 20.04)

- Python 3.7-3.10 (tested on 3.9)

- Anaconda / Miniconda 

```curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh```

```bash Miniconda3-latest-Linux-x86_64.sh```

### Installation

Download /ContentRepresentation and navigate to the folder

```conda create --name content_repr python=3.9```

```conda activate content_repr```

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/```

```conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0```

```pip install --upgrade pip```

```pip install -r requirements.txt```

```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```

```cd data/input && curl http://cs231n.stanford.edu/tiny-imagenet-200.zip```

```unzip -qq 'tiny-imagenet-200.zip' && cd ../..```

```python3 generate_content_representations.py --help```


### To generate a content representation of a single image:

```python3 generate_content_representations.py -f IMAGE_FILE -x 512 -y 512 -i 600```

(you may experiment with other arguments and argument values)


### To generate a training dataset from Tiny ImageNet:

```python3 generate_content_representations.py -d IMAGES_DIR -x 64 -y 64 -e```

(you may experiment with other arguments and argument values)


## Training a Content Representation ControlNet
This section shows the setup for training a Content Representation ControlNet using Google Colaboratory. 

### Requirements

Google Drive (13 GB + space initially, 7.7 GB can be freed after)

or

Nvidia GPU with 8GB+ VRAM and modifications to the provided code.

### Installation

```git clone https://github.com/btancovski/ControlNet-March2023```

(or alternatively, using the latest version from https://github.com/lllyasviel/ControlNet - may require code adaptations)

```curl https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt```

Upload the full project and the model to Google Drive (the model goes into the models folder)

Copy the files in the ```/ControlNet``` to the Google Drive folder (replacing existing files)

### Training

Open ```colab_train.ipynb```, enable GPU acceleration and execute the first cell (the runtime will disconnect after this - this is expected) and then every other cell in order. The outputs of the model will be saved in a newly created folder /image_log.
