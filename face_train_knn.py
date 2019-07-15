from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

from os.path import join as pjoin
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib



from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

def load_and_align_data(image, image_size, margin, gpu_memory_fraction):

 
    img = image
   
    img_size = np.asarray(img.shape)[0:2]
  
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

  
    if len(bounding_boxes) < 1:
        return 0,0,0
    else:
        crop=[]
        det=bounding_boxes

        det[:,0]=np.maximum(det[:,0], 0)
        det[:,1]=np.maximum(det[:,1], 0)
        det[:,2]=np.minimum(det[:,2], img_size[1])
        det[:,3]=np.minimum(det[:,3], img_size[0])

        # det[:,0]=np.maximum(det[:,0]-margin/2, 0)
        # det[:,1]=np.maximum(det[:,1]-margin/2, 0)
        # det[:,2]=np.minimum(det[:,2]+margin/2, img_size[1])
        # det[:,3]=np.minimum(det[:,3]+margin/2, img_size[0])

        det=det.astype(int)

        for i in range(len(bounding_boxes)):
            temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
            aligned=misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
        crop_image=np.stack(crop)
            
        return det,crop_image,1



def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(data_dir):

  
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)       
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]         

       
        data[guy] = curr_pics      
    return data


def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      



    if gray.ndim == 2:
        img = to_rgb(gray)
    return img


model_dir='./20170512-110547'

with tf.Graph().as_default():
    with tf.Session() as sess:
       
        facenet.load_model(model_dir)

       
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

      
        data=load_data('./train_dir/')

   
        keys=[]
        for key in data:
            keys.append(key)
      

        train_x=[]
        train_y=[]

     
        for x in data[keys[0]]:
            _,images_me,i = load_and_align_data(x, 160, 44, 1.0)
            if i:
                feed_dict = { images_placeholder: images_me, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict) 
                print(type(emb))

                for xx in range(len(emb)):
                    print(type(emb[xx,:]),emb[xx,:].shape)
                    train_x.append(emb[xx,:])       
                    train_y.append(0)
        print(len(train_x))


        for y in data[keys[1]]:
            _,images_others,j = load_and_align_data(y, 160, 44, 1.0)
            if j:
                feed_dict = { images_placeholder: images_others, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                for xx in range(len(emb)):
                    print(type(emb[xx,:]),emb[xx,:].shape)
                    train_x.append(emb[xx,:])
                    train_y.append(100)
        print(len(train_x))


          
train_x=np.array(train_x)
print(train_x.shape)
train_x=train_x.reshape(-1,128)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  

classifiers = knn_classifier 

model = classifiers(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  
    
#save model
joblib.dump(model, './models/knn_classifier.model')

model = joblib.load('./models/knn_classifier.model')
predict = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 






