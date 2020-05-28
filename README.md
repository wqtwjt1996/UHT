# UHT
The pytorch re-implementation of paper: A method for detecting text of arbitrary shapes in natural scenes that improves text spotting. 

![image](https://github.com/wqtwjt1996/UHT/blob/master/vis.jpg)
- Until now, only Total-text and COCO-Text datasets are implemented. 
- Only ResNet-50 backbone is implemented.
- Parts of code, such as data augmentation refered: https://github.com/princewang1994/TextSnake.pytorch
## How to setting the parameters?
Please go to config.py to setting them.
## My running environment
```
python 3.7
Pytorch 1.4.0
OpenCV 3.4.2
CUDA 10.2
Ubuntu 18.04.2
```
## Now let's try the code!
To train the code, running:
```
python train.py
```
If you want to pretrain using SynthText dataset, setting "dataset_name" in config.py as: synth_text
If you want to fine-tune using Total-Text or COCO-Text dataset, setting "dataset_name" in config.py as corresponding dataset name.
Remember downloading the datasets first. Here are some links that might help you:
- https://www.robots.ox.ac.uk/~vgg/data/scenetext/
- https://github.com/cs-chan/Total-Text-Dataset
- https://vision.cornell.edu/se3/coco-text-2/

To test the model, running:
```
python eval.py
```

We recommend these following evaluation code:
- https://github.com/cs-chan/Total-Text-Dataset/tree/master/Evaluation_Protocol

Default setting tr = 0.8 and tp = 0.4 is implemented in our experiments.
- https://github.com/liuheng92/OCR_EVALUATION

You can try our trained model: "eg_model.pth" from:
- https://drive.google.com/file/d/1eCaJXontrOaH67rpxGh9-R-GAkaOXoCF/view?usp=sharing

More updates are coming soon. :)
