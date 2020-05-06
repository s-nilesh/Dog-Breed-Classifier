# Dog-Breed-Classifier - CNN and transfer learning in PyTorch

Aim is to build an ML workflow (consisting of 3 models) that could be used within a web app to process real world, user-supplied images in the following way:
- ***Human Face detector*** - If given an image, model outputs *True if Human face* is detected, else returns False 
- ***Dog detector*** - If given an image, model outputs *True if dog* is detected else outputs False.
- ***Dog breed detector*** - If given an image, model predicts the *dog breed* associated with that that image. If it detects dog, it will predict dog breed, if human face is detected, it predicts the dog breed that resembles with the human face.

## Project Overview
Humans are excellent at vision, our brain is very powerful in visual recognition. Given a dog, one can easily detect it’s breed the only condition is you be aware of all the dog breeds on this planet! Now this becomes a quite challenging task for a normal human. Consider you love a specific dog breed(say, labrador) and want to adopt a dog of the same breed, you go to shop and bring home your lovely new friend. How do you know if that’s the correct dog breed you have got?? Many times it becomes hard for humans to identify the dog’s breed. For example how about classifying the two dog images given below.<br>
<img src='imagesForREADME/Screenshot from 2020-05-05 15-48-19.png'>

So, this becomes a tough job. Here is when the need for Machine Learning/Deep Learning comes into the picture. Computer Vision helps you build machine learning models where you train a model to recognize the dog breed! This makes your job easy! CNN was a huge invention in deep learning, this gave a boost to lots of applications in visual recognition. I will walk you through how to build a CNN from scratch using Pytorch and leverage some of the popular CNN architectures for our image classification task!<br>

## Dataset
Dataset Needed for the project can be found in these two folder(hyperlinked):
- Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

***Distribution of the data***:
- Total number of human **face images: 13233**
- Total number of human **face folders: 5749**
- Total numner of folders in 'dog_images:' 3
- Folders in 'dog_images': train,test,valid
- Total folders(**breed classes**) in 'train, test, valid' **133**
- Total images in /dog_images/**train : 6680**
- Total images in /dog_images/**test : 836**
- Total images in /dog_images/**valid : 835**

## Human face detector
We have used [OpenCV's implementation of Haar Cascade Classifier](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) to detect human faces in the given image.<br>

> Sample output given by Haar Cascade Classifier

<img src='imagesForREADME/Screenshot from 2020-05-06 09-21-28.png'>

*98.74%* of total images in human dataset are correctly detected human face images and *10.83%* of total dog images are incorrectly detected as human face image<br>

## Dog detector 
Used VGG16 for transfer learning. <br>
This model predicts dog images with *100% accuracy*!<br>
> Sample predictions made by this model:

<img src='imagesForREADME/Screenshot from 2020-05-06 11-22-48.png'>
<img src='imagesForREADME/Screenshot from 2020-05-06 11-23-00.png'>

## Dog Breed Classifier
This is a multi-class classification task, made one 6-layer [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) from scratch and then used [ResNet-101](https://arxiv.org/abs/1512.03385) for transfer learning.<br>

#### CNN Architecture:
`**Layer-1**

(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) -> size of the image will reduce by a factor of 2, as we have applied stride of (2,2) and padding of (1,1) for (3,3) kernel [(224,224) -> (112,112)]
Activation: RELU
(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) size of image will be reduced by factor of 2 [(112,112) -> (56,56)]
Output shape for each image after his layer is expected to be (56,56,32)

**Layer-2**

(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) -> size of the image will reduce by a factor of 2, as we have applied stride of (2,2) and padding of (1,1) for (3,3) kernel [(56,56) -> (28,28)]
Activation: RELU
(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) size of image will be reduced by factor of 2 [(28,28) -> (14,14)]
Output shape for each image after his layer is expected to be (14,14,64)

**Layer-3**

(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) -> size of the image remains the same here as we have applied stride of (1,1) and padding of (1,1) for (3,3) kernel [(14,14) -> (14,14)]
Activation: RELU
(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) size of image will be reduced by factor of 2 [(14,14) -> (7,7)]
Output shape for each image after his layer is expected to be (7,7,128)

Downsized the image from (224,224) to (7,7) increasing the image depth from 3 to 128, we can now use this as features from images to build our breed classifier

Flattening

We flatten the image to get a 25088 sized vector or this can be imagined as 25088 hidden nodes layer with is further connected to fc4
I have then applied dropout of 0.25 which keeps deactivates 25% of neurons of this layer

**Layer-4**

(fc4): Linear(in_features=6272, out_features=2048, bias=True) -> We have extracted 6272 features from each image and now I have built a Fully connected layer with 512 hidden nodes.
(batch_norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
Activation: RELU
Applied dropout of 0.25 which keeps deactivates 25% of neurons of this layer

**Layer-5**

(fc5): Linear(in_features=2048, out_features=512, bias=True)
Activation: RELU
Applied dropout of 0.25 which keeps deactivates 25% of neurons of this layer

**Layer-6**

(fc5): Linear(in_features=512, out_features=133, bias=True) Fully-connected layer with 133 hidden nodes(number of classes)
Thus finally extracted 133 sized vector, one for each data point
This 133 sized vector is then used to predict classses`


#### Transfer Learning
Used pre-trained ResNet-101 model by removing its last fully-connected layer and attaching our own fully-connected layer to get 133 shped vector output!<br>

> Model evaluation:<br>
***Precision*** : 0.804<br>
***Recall*** : 0.782<br>

## Dog App
Finally combining models to get the required workflow:,br>

> Sample Results on my own data(user supplied images):

<img src='imagesForREADME/Screenshot from 2020-05-06 11-34-10.png'>
<img src='imagesForREADME/Screenshot from 2020-05-06 11-34-56.png'>
<img src='imagesForREADME/Screenshot from 2020-05-06 11-34-18.png'>
<img src='imagesForREADME/Screenshot from 2020-05-06 11-33-47.png'>
<img src='imagesForREADME/Screenshot from 2020-05-06 11-33-58.png'>
<img src='imagesForREADME/Screenshot from 2020-05-06 11-34-27.png'>

Now this dog_app can be used to make a web application or a phone application to get predictions!
