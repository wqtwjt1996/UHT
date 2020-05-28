# UHT
The pytorch re-implementation of paper: A method for detecting text of arbitrary shapes in natural scenes that improves text spotting. 
- Until now, only Total-text and COCO-Text datasets are implemented. 
- Only ResNet-50 backbone is implemented.
- Parts of code, such as data augmentation refered: https://github.com/princewang1994/TextSnake.pytorch
## How to setting the parameters?
Please go to config.py to seeting them.
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
To test the model, running:
```
python eval.py
```

