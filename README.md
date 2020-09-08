Scene Parsing also known as scene segmentation expects to segment an entire image in multiple objects which acts as crucial component in many higher level tasks such as scene understanding, Autonomous vehicles and satellite image analysis. A higher level understanding of the image is required if one wants to perform scene segmentation. The algorithm should not only understand the objects that are present but also the pixels that correspond to the object.
In this paper, we will how we used deep convolutional neural network to do image segmentation. We will discuss the whole pipeline – from preparing the data to building the model.
There were various steps discussed in the paper, but the approach used by us was a bit different and as such the results obtained by us were not similar to the ones obtained by the authors.

<h3>DATASET</h3> 

The Dataset on which we trained and tested the network was taken from the PASCAL Visual object classes challenge 2012 (VOC2012). The dataset contains a total of 11540 images of 20 different class labels. The 20 different classes are – ‘person’ , ‘bird’, ‘cat’, ‘dog’, ‘horse’, ‘sheep’, ‘aeroplane’, ‘bicycle’, ‘boat’, ‘bus’, ‘car’, ‘motorbike’, ‘train’, ‘bottle’, ‘chair’, ‘dining table’, ‘potted plant’, ‘sofa’ and ‘TV’.
Out of this large dataset, we selected a small subset of 367 images of 2007 only. This was done because of the computation limitation of our system. As this is not a classification task, the balance of the dataset was not a important parameter when selecting images and was not taken into consideration as it does not affect the performance of the Model.
A sample image of the dataset is shown if Fig. 1.

