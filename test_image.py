from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

#import image_pb2
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from facenet.packages import facenet, detect_face


#if 1 == True:
#    match_threshold=0.2
#    img_path = './test_images/test0.jpg'
#    frame = cv2.imread(img_path,0)

##when passed an image frame (numpy array or cv::Mat format), this function detects faces and attempts to classify them
def test_image(frame, match_threshold=0.2):
    mypath = os.path.dirname(__file__)
    modeldir =  mypath + '/model'
    classifier_filename = mypath + '/classifier/classifier.pkl'
    npy=''

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            image_size = 182
            input_image_size = 160
            

#            print('Loading feature extraction model')
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, HumanNames) = pickle.load(infile)

            HumanNames.pop()

#            print('Start Recognition!')

            find_results = []
            result_names = []
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            
#            print('Face Detected: %d' % nrof_faces)
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        result_names.append('Too Close')
                        bb[i][0] = max(bb[i][0], 1)
                        bb[i][1] = max(bb[i][1], 1)
                        bb[i][2] = min(bb[i][2], len(frame[0])-1)
                        bb[i][3] = min(bb[i][3], len(frame)-1)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                        cv2.putText(frame, "Image Edge Overlap", (bb[i][0]+20, bb[i][1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 0, 255), thickness=1, lineType=2)
                        continue

                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                        interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                    
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    
                    #print(predictions)
                    best_class_indices = np.argmax(predictions, axis=1)
                    # print(best_class_indices)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    #print(best_class_probabilities)
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    
                    if best_class_probabilities > match_threshold:
                        result_names.append(HumanNames[best_class_indices[0]])
                    else:
                        result_names.append('Unknown Person')

                    
                    cv2.putText(frame, result_names[i], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 0, 255), thickness=1, lineType=2)
                    
            else:
                bb = np.zeros((1,4), dtype=np.int32)
                cv2.putText(frame, "No Faces Detected", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 0, 255), thickness=1, lineType=2)

            return frame, nrof_faces, result_names, bb

