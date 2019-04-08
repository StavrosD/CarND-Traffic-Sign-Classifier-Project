# **Traffic Sign Recognition** 

## Writeup

### German Traffic Signs Classification project

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *(32, 32, 3)*
* The number of unique classes/labels in the data set is *32*

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between the images

![Traffic Signs Histogram][./histogram.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the model can train faster with one color channel instead of two.

Then I normalized the image so each value will be within [-1,1]

Here is an example of a traffic sign image before and after preprocessing.

![Before preprocessing][./original_sign.png]
![After preprocessing][./preprocessed_image.png]

**Preprocessing improvements**

I did not have enough time for the improvements I wanted to implement.

I have planned to generate additional data because some traffic signs have more images than other traffic signs. Creating equal images for each traffic sign would increase the training accuracy.

Also adaptive Contrast Limited Adaptive Histogram Equalization (CLAHE) would further increase the training accuracy but it was so slow on my machine that I disabled it, I could not work with CLAHE enabled. 
Tha code for CLAHE is: 
for i in range(img.shape[0]):
	img[i] = exposure.equalize_adapthist(img[i])

To add more data to the the data set, I would use the following techniques:
- store each image color average
- transform (rotate, scale, move, mirror) the original image. The margins will be filled with the color average
- I would repeat the same process until there are the same images for each traffic sign.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Flatten		      	| outputs 400									|		|
| Fully connected		| outputs 120									|
| Fully connected		| outputs 84									|
| Fully connected		| outputs 43									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an epoch equal to 30, a rate equal to 0.005 and a batch size equal to 128.

The optimizer I used is the *adam optimizer*.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **0.997**
* validation set accuracy of  **0.945**
* test set accuracy of **0.921**

I started with the LaNet example.
With the initial architecture, the required validation accuracy was lower than the required (0.93).
I modified the algorithm by intuition, I tested different steps until I got a training value greater than 0.93.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][./1.png] ![alt text][./2.png] ![alt text][./3.png] 
![alt text][./4.png] ![alt text][./5.png]


The second image might be difficult to classify because it is too bright.
The third image might be difficult to classify because of the noise, it looks like there is some plant on the traffic sign.
The fifth image might be difficult to classify because it is only partially displayed.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (80km/h)	| Speed limit (80km/h)							| 
| General caution		| General caution    							|
| Dangerous curve to the left	| Dangerous curve to the left			|
| Children crossing   	| Children crossing					 			|
| Road narrows on the right	| Traffic signals     						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is retlatevely sure that this is a Speed limit (80km/h) sign (probability of 0.76), and the image does contain a Speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.76         			| Speed limit (80km/h)		  					| 
| 0.13     				| Speed limit (30km/h)							|
| 0.06					| Speed limit (70km/h)							|
| 0.03	      			| Speed limit (50km/h)				 			|
| 0.02				    | Turn right ahead (2%)							|
![alt text][./1.png] 



For the first image, the model is absulutely sure that this is a General caution sign (probability of 1.00), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution		  						| 
| 0.00     				| Priority road									|
| 0.00					| Children crossing								|
| 0.00	      			| Traffic signals 					 			|
| 0.00				    | Right-of-way at the next intersection			|
![alt text][./2.png] 

For the third image, the model is absulutely sure that this is a Dangerous curve to the left sign (probability of 1.00), and the image does contain a Dangerous curve to the left sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Dangerous curve to the left					| 
| 0.00     				| Slippery road									|
| 0.00					| No passing									|
| 0.00	      			| Dangerous curve to the right		 			|
| 0.00				    | Bicycles crossing								|
![alt text][./3.png] 

For the fourth image, the model is absulutely sure that this is a Children crossing sign (probability of 1.00), and the image does contain a Children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children crossing								| 
| 0.00     				| Priority road									|
| 0.00					| Ahead only									|
| 0.00	      			| Bicycles crossing					 			|
| 0.00				    | Go straight or right							|
![alt text][./4.png] 

For the fifth image, the model is relative sure that this is a End of speed limit (80km/h sign (probability of 0.98), but the image does contain a Road narrows on the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| End of speed limit (80km/h					| 
| 0.00     				| End of all speed and passing limits			|
| 0.00					| Go straight or right							|
| 0.00	      			| nd of no passing 					 			|
| 0.00				    | Roundabout mandatory							|
![alt text][./5.png]

### Performance considerations
#### 1. Speed up using CUDA
I enabled GPU support on my PC and I used CUDA for training.
I configured the session so it will display on the log if GPU or CPU is used. 
``` python
config = tf.ConfigProto()               # configure tensorflow session
config.log_device_placement=True        # log CPU or GPU is used
config.gpu_options.allow_growth = True  # allow dynamically allocate memory to prevent an error on my system, may not be needed on other systems
sess = tf.Session(config=config)

with sess:
```

#### 2. Speed up using parallel threads
Multiple threads can be used for faster performance

```
def normalize(img,result,exit_message):
    result = RGB2NORMALIZED_GREY(img)
    print(exit_message)
    return result
Normalized_X_train=[]
Normalized_X_valid=[]
Normalized_X_test=[]
p1 = Thread(target = normalize,args=(X_train,Normalized_X_train,"X_train normalized"))
p2 = Thread(target = normalize,args=(X_valid,Normalized_X_valid, "X_valid normalized"))
p3 = Thread(target = normalize,args=(X_test,Normalized_X_test, "X_test normalized"))
p1.start()
p2.start()
p3.start()
p1.join()
p2.join()
p3.join()
```

#### 3. Speed up using multiplication instead of division
For CUDA GPU, multiplication is more than 400% faster than division (8 divisions per CPU cycle vs 1.6 division/cycle)

[NVIDIA OpenCL ProgrammingGUide](http://www.nvidia.com/content/cudazone/download/OpenCL/NVIDIA_OpenCL_ProgrammingGuide.pdf)

 for a 6th generation intel CPU, multiplication takes 1~2 cycles while division takes 24~90 cycles for a 64bit number
 multiplication is up to 9000% faster
 
 [Lists of instruction latencies, throughputs and micro-operation breakdowns for Intel, AMD
and VIA CPUs](http://www.agner.org/optimize/instruction_tables.pdf)

 I modified the normalization equation so it will do the same calculations without division, the devision is pre-calculated
 (pixel-128)/128 = pixel/128-128/128 = **pixel*0.0078125 - 1 **

```
def RGB2NORMALIZED_GREY(img):
    img=np.dot(img[...,:3],[0.4,0.2,0.4])    # most traffic signs have red or blue color, no images on the dataset have green color. I assign larger weights to the RED and BLUE color. It also helps to filter out the trees and the grass.  
#    for i in range(img.shape[0]):
#        img[i] = exposure.equalize_adapthist(img[i])
    img = img*0.0078125-1
    img=np.reshape(img,img.shape+(1,))
    return img
```
#### 4. Speed up using parallel processes
Parallell processes could further enchance the performance of the scripts, however palallel processing is not supported by jupyter on windows sytem so I could not implement it.




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


