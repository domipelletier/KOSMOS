# An AI for KOSMOS
## 2021 Brest Ocean Hackathon

<p align="center">
<img src="KOSMOS.jpg" style="vertical-align:middle" width="500" height='600' class='center' alt='logo'>
</p>

### Installation

pre-requisites:
- miniconda3
- python3

For the whole project, you need to create and use a virtual environment.  
You can use any version of python3 superior to 3.6. We recommend to use at least 3.9.

You can use any virtual environment manager, we personnally use miniconda3.

```console
conda create -n kosmos python=3.9
conda activate kosmos

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Testing

Both files `opencv_image.py` and `opencv_video.py` are provided to test classification
and detection models, respectively on a single image or a video.

### Training

To train the model we used for detection, you need to launch the training script.  
You can find it in the `detection` folder.

### Classification

You can find the classifier model in the `classification` folder.
