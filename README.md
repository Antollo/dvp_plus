# Intro

This repository contains the implementation of the DVP+ method presented in "Generative Learning of Differentiable Object Models for Compositional Interpretation of Complex Scenes", a manuscript submitted to NeurIPS'25 [https://openreview.net/forum?id=7fyjBHx5Ac](https://openreview.net/forum?id=7fyjBHx5Ac) alongside with the MDS-HR benchmark introduced therein.

# Installation

We assume python 3.10 is installed.

```sh
python -m venv .venv
source .venv/bin/activate
pip install fvcore ipykernel tensorboard transformers ipywidgets seaborn shapely opencv-python scikit-learn-extra scikit-image numpy==1.26.4
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html
pip install ./torch_earcut/ ./lcmr/ ./lcmr-ext/

# MONet
cd notebooks
git clone https://github.com/addtt/object-centric-library
mv object-centric-library object_centric_library

# LIVE
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom cairosvg beautifulsoup4
git clone https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization
cp LIVE_runner.sh ./LIVE-Layerwise-Image-Vectorization/LIVE/
cd LIVE-Layerwise-Image-Vectorization/DiffVG
git submodule update --init --recursive
python setup.py install
cd ../..

# Running LIVE
cd LIVE-Layerwise-Image-Vectorization/LIVE
./LIVE_runner.sh
```

# Getting the Dataset

You can either generate the dataset or download a pre-existing version:
*   **To generate:** Execute the `notebooks/dataset_generator.ipynb` notebook.
*   **To download:** Access it via OSF: [https://osf.io/pwhd7/?view_only=60ad0f39814b49eda85a61a86f60fae0](https://osf.io/pwhd7/?view_only=60ad0f39814b49eda85a61a86f60fae0)

# Model Training

For each trainable DVP+ configuration, a specific training notebook is provided within the `notebooks/` directory.

# Running Experiments

The `notebooks/test.ipynb` notebook allows you to reproduce all experiments reported in the paper.