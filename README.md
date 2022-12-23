# Inbetweening

## Installation and usage (tested in Google Colab)

### Installation

* Create conda environments named env7 and env9 with python3.7 and python3.9 respectively

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local
conda create --name env7 python=3.7 -y
conda create --name env9 python=3.9 -y
```

* Follow instructions from [Sketch2Pose](https://github.com/kbrodt/sketch2pose) and [Robust Motion Inbetweening](https://github.com/jihoonerd/Robust-Motion-In-betweening) repos to clone and set up them

* Put pairs of input images in separate dirs in the `input` directory

### Sketch2Pose

* Change directory to `sketch2pose`

```
cd sketch2pose
```

* Apply patches

```
patch scripts/prepare.sh < ../patches/prepare_patch.diff
patch patches/selfcontact.diff < ../patches/selfcontact_patch_patch.diff
patch patches/smplx.diff < ../patches/smplx_patch_patch.diff
patch patches/torchgeometry.diff < ../patches/torchgeometry_patch_patch.diff
patch src/pose.py < ../patches/pose_patch.diff
```

* Set up Sketch2Pose

```
eval "$(conda shell.bash hook)"
conda activate env9

bash ./scripts/download.sh

pip install -U pip setuptools

pip install \
    torch \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/"${extra}"

pip install -r requirements.txt

v=$(python -c 'import sys; v = sys.version_info; print(f"{v.major}.{v.minor}")')
for p in patches/*.diff; do
    patch -d/ -p0 < <(sed "s/python3.10/python${v}/" "${p}")
done
conda deactivate
```

* Run Sketch2Pose

```
eval "$(conda shell.bash hook)"
conda activate env9
python src/generate_poses.py \
       --input-path "../input"
conda deactivate
```

* Change directory to root

```
cd ..
```

### Retargeting to bvh

* Install bpy and other packages

```
eval "$(conda shell.bash hook)"
conda activate env7
wget -O ./bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
pip install ./bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
pip install mathutils==2.81.2
pip install opencv-python
pip install argparse
conda deactivate
```

* Do retargeting

```
eval "$(conda shell.bash hook)"
conda activate env7
python smplx_to_bvh.py
conda deactivate
```

### Robust Motion Inbetweening

* Change directory to `Robust-Motion-In-betweening`

```
cd Robust-Motion-In-betweening
```

* Apply patches

```
patch rmi/data/lafan1_dataset.py < ../patches/lafan1_patch.diff
patch test.py < ../patches/test_patch.diff
patch requirements.txt < ../patches/requirements_patch.diff
patch PyMO/pymo/parsers.py < ../patches/parsers_patch.diff
patch config/config_base.yaml < ../patches/config_patch.diff
```

* Set up Robust Motion Inbetweening

```
eval "$(conda shell.bash hook)"
conda activate env9
pip install -r requirements.txt
cd PyMO
python setup.py install
cd ..
conda deactivate
```

* Run Robust Motion Inbetweening

```
eval "$(conda shell.bash hook)"
conda activate env9
python test.py
conda deactivate
```

* Change directory to root

```
cd ..
```

### Processing results

* Collect Robust Motion Inbetweening output and transform it into bvh files in the `output` directory

```
eval "$(conda shell.bash hook)"
conda activate env7
python jsons_to_bvh.py
conda deactivate
```

## Reference
```
@article{brodt2022sketch2pose,
    author  = {Kirill Brodt and Mikhail Bessmeltsev},
    title   = {Sketch2Pose: Estimating a 3D Character Pose from a Bitmap Sketch},
    journal = {ACM Transactions on Graphics},
    year    = {2022},
    month   = {7},
    volume  = {41},
    number  = {4},
    doi     = {10.1145/3528223.3530106},
}
```
```
@article{harvey2020robust,
    author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
    title     = {Robust Motion In-Betweening},
    booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
    publisher = {ACM},
    volume    = {39},
    number    = {4},
    year      = {2020}
}
```
