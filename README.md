# ViT_CIFAR10

In this repository, I have implemented ViT, which was suggested in "AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE". 
CIFAR10 data are used for training and testing.

ViT network is defined in "ViT.py" from a scrach. 
Training and testing were done in "ViT_CIFAR10.py".

The training requires a lot of time. Therefore, I set the depth of ViT to 1. 
Nevertheless, the training for ViT required more time than a very simple CNN, though the performance was worse than the simple CNN.
