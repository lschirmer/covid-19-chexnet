# covid-19-chexnet
covid-19 detection with Keras and Tensorflow.

We use ChexNet to automatically detect COVID-19 in a hand-created X-ray image dataset.

In this application we use transfer learning from a pre-trained network (ChexNet) made available by Bruce Chou [CheXNet-Keras](https://github.com/brucechou1983/CheXNet-Keras). Also, part of the application was built following the tutorial available [here](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/) by Adrian Rosebrock.

# ChexNet
ChexNet is a deep learning algorithm that can detect and localize 14 kinds of diseases from chest X-ray images. As described in the paper, a 121-layer densely connected convolutional neural network is trained on ChestX-ray14 dataset, which contains 112,120 frontal view X-ray images from 30,805 unique patients. We adapted the model to detect and classify lung infections caused by coronavirus.

# Dataset
The COVID-19 X-ray image dataset, available [here](https://github.com/ieee8023/covid-chestxray-dataset) was curated by Dr. Joseph Cohen, a postdoctoral fellow at the University of Montreal.

# Requirements
1. Tensorflow
2. Keras
3. Scikit-learn
4. OpenCV
5. imutils

# Disclaimer!!!!
COVID-19-chexnet detection is for educational purposes only. It is not meant to be a reliable, highly accurate COVID-19 diagnosis system, nor has it been professionally or academically vetted. This is not a scientifically rigorous study. 

As said by Adrian Rosebrock, one of the biggest limitations of the method discussed here is data. We simply don’t have enough data to train a COVID-19 detector reliably. The method covered here is only for educational purposes.

