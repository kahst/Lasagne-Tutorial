#########################################################
#                LASAGNE TUTORIAL PART 1                #
#                    BY STEFAN KAHL                     #
# VISIT: http://medien.informatik.tu-chemnitz.de/skahl/ #
#########################################################

import os
from sklearn.utils import shuffle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates

import theano
import theano.tensor as T

from timer import Timer

################### DATASET HANDLING ####################
DATASET_PATH = 'dataset'
def parseDataset():

    #we use subfolders as class labels
    classes = [folder for folder in os.listdir(DATASET_PATH)]

    #now we enlist all image paths
    images = []
    for c in classes:
        images += ([os.path.join(DATASET_PATH, c, path) for path in os.listdir(os.path.join(DATASET_PATH, c))])

    #shuffle image paths
    images = shuffle(images, random_state=42)

    #we want to use a 15% validation split
    vsplit = int(len(images) * 0.15)
    train = images[:-vsplit]
    val = images[-vsplit:]

    #show some stats
    print "CLASS LABELS:", classes
    print "TRAINING IMAGES:", len(train)
    print "VALIDATION IMAGES:", len(val)

    return classes, train, val

#parse dataset
CLASSES, TRAIN, VAL = parseDataset()

#################### BATCH HANDLING #####################
def loadImageAndTarget(path):

    #here we open the image and scale it to 64x64 pixels
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    
    #OpenCV uses BGR instead of RGB, but for now we can ignore that
    #our image has the shape (64, 64, 3) but we need it to be (3, 64, 64)
    img = np.transpose(img, (2, 0, 1))
    
    #we want to use subfolders as class labels
    label = path.split("/")[-2]

    #we need to get the index of our label from CLASSES
    index = CLASSES.index(label)

    #allocate array for target
    target = np.zeros((5), dtype='float32')

    #we set our target array = 1.0 at our label index, all other entries remain zero
    #Example: if label = dog and dog has index 2 in CLASSES, target looks like: [0.0, 0.0, 1.0, 0.0, 0.0]
    target[index] = 1.0

    #we need a 4D-vector for our image and a 2D-vector for our targets
    #we can adjust array dimension with reshape
    img = img.reshape(-1, 3, 64, 64)
    target = target.reshape(-1, 5)

    return img, target

#a reasonable size for one batch is 128
BATCH_SIZE = 128
def getDatasetChunk(split):

    #get batch-sized chunks of image paths
    for i in xrange(0, len(split), BATCH_SIZE):
        yield split[i:i+BATCH_SIZE]

def getNextImageBatch(split=TRAIN):    

    #allocate numpy arrays for image data and targets
    #input shape of our ConvNet is (None, 3, 64, 64)
    x_b = np.zeros((BATCH_SIZE, 3, 64, 64), dtype='float32')
    #output shape of our ConvNet is (None, 5) as we have 5 classes
    y_b = np.zeros((BATCH_SIZE, 5), dtype='float32')

    #fill batch
    for chunk in getDatasetChunk(split):        
        ib = 0
        for path in chunk:
            #load image data and class label from path
            x, y = loadImageAndTarget(path)

            #pack into batch array
            x_b[ib] = x
            y_b[ib] = y
            ib += 1

        #instead of return, we use yield
        yield x_b[:len(chunk)], y_b[:len(chunk)]

################## BUILDING THE MODEL ###################
def buildModel():

    #this is our input layer with the inputs (None, dimensions, width, height)
    l_input = layers.InputLayer((None, 3, 64, 64))

    #first convolutional layer, has l_input layer as incoming and is followed by a pooling layer
    l_conv1 = layers.Conv2DLayer(l_input, num_filters=64, filter_size=3)
    l_pool1 = layers.MaxPool2DLayer(l_conv1, pool_size=2)

    #second convolution (l_pool1 is incoming), let's increase the number of filters
    l_conv2 = layers.Conv2DLayer(l_pool1, num_filters=128, filter_size=3)
    l_pool2 = layers.MaxPool2DLayer(l_conv2, pool_size=2)

    #third and final convolution, even more filters
    l_conv3 = layers.Conv2DLayer(l_pool2, num_filters=256, filter_size=3)
    l_pool3 = layers.MaxPool2DLayer(l_conv3, pool_size=2)

    #our cnn contains 3 dense layers, one of them is our output layer
    l_dense1 = layers.DenseLayer(l_pool3, num_units=1024)
    l_dense2 = layers.DenseLayer(l_dense1, num_units=1024)

    #the output layer has 5 units which is exactly the count of our class labels
    #it has a softmax activation function, its values represent class probabilities
    l_output = layers.DenseLayer(l_dense2, num_units=5, nonlinearity=nonlinearities.softmax)

    #let's see how many params our net has
    print "MODEL HAS", layers.count_params(l_output), "PARAMS"

    #we return the layer stack as our network by returning the last layer
    return l_output

NET = buildModel()

#################### LOSS FUNCTION ######################
def calc_loss(prediction, targets):

    #categorical crossentropy is the best choice for a multi-class softmax output
    l = T.mean(objectives.categorical_crossentropy(prediction, targets))
    
    return l

#theano variable for the class targets
#this is the output vector the net should predict
targets = T.matrix('targets', dtype=theano.config.floatX)

#get the network output
prediction = layers.get_output(NET)

#calculate the loss
loss = calc_loss(prediction, targets)

################# ACCURACY FUNCTION #####################
def calc_accuracy(prediction, targets):

    #we can use the lasagne objective categorical_accuracy to determine the top1 accuracy
    a = T.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
    
    return a

accuracy = calc_accuracy(prediction, targets)

####################### UPDATES #########################
#get all trainable parameters (weights) of our net
params = layers.get_all_params(NET, trainable=True)

#we use the adam update
#it changes params based on our loss function with the learning rate
param_updates = updates.adam(loss, params, learning_rate=0.00001)

#################### TRAIN FUNCTION ######################
#the theano train functions takes images and class targets as input
#it updates the parameters of the net and returns the current loss as float value
#compiling theano functions may take a while, you might want to get a coffee now...
print "COMPILING THEANO TRAIN FUNCTION...",
train_net = theano.function([layers.get_all_layers(NET)[0].input_var, targets], loss, updates=param_updates)
print "DONE!"

################# PREDICTION FUNCTION ####################
#we need the prediction function to calculate the validation accuracy
#this way we can test the net after training
#first we need to get the net output
net_output = layers.get_output(NET)

#now we compile another theano function; this may take a while, too
print "COMPILING THEANO TEST FUNCTION...",
test_net = theano.function([layers.get_all_layers(NET)[0].input_var, targets], [net_output, loss, accuracy])
print "DONE!"

##################### STAT PLOT #########################
plt.ion()
label_added = False
def showChart(epoch, t, v, a):

    #x-Axis = epoch
    e = range(0, epoch + 1)

    #loss subplot
    plt.subplot(211)
    plt.plot(e, train_loss, 'r-', label='Train Loss')
    plt.plot(e, val_loss, 'b-', label='Val Loss')
    plt.ylabel('loss')

    #show labels only once
    global label_added
    if not label_added:
        plt.legend(loc='upper right', shadow=True)
    label_added = True

    #accuracy subplot
    plt.subplot(212)
    plt.plot(e, val_accuracy, 'g-')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    #show
    plt.show()
    plt.pause(0.5)
    

###################### TRAINING #########################
print "START TRAINING..."
timer = Timer()
train_loss = []
val_loss = []
val_accuracy = []
for epoch in range(0, 100):

    #start timer
    timer.tic()

    #iterate over train split batches and calculate mean loss for epoch
    t_l = []
    for image_batch, target_batch in getNextImageBatch():

        #calling the training functions returns the current loss
        l = train_net(image_batch, target_batch)
        t_l.append(l)

    #we validate our net every epoch and pass our validation split through as well
    v_l = []
    v_a = []
    for image_batch, target_batch in getNextImageBatch(VAL):

        #calling the test function returns the net output, loss and accuracy
        prediction_batch, l, a = test_net(image_batch, target_batch)
        v_l.append(l)
        v_a.append(a)

    #stop timer
    timer.toc()

    #calculate stats for epoch
    train_loss.append(np.mean(t_l))
    val_loss.append(np.mean(v_l))
    val_accuracy.append(np.mean(v_a))

    #print stats for epoch
    print "EPOCH:", epoch,
    print "TRAIN LOSS:", train_loss[-1],
    print "VAL LOSS:", val_loss[-1],
    print "VAL ACCURACY:", (int(val_accuracy[-1] * 1000) / 10.0), "%",
    print "TIME:", (int(timer.diff * 10) / 10.0), "s"

    #show chart
    showChart(epoch, train_loss, val_loss, val_accuracy)

print "TRAINING DONE!"
