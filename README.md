# Description
This is an implementation of facenet derived from:
https://github.com/davidsandberg/facenet

This repo carries an adaptation of some automation from [AISangam's repo](https://github.com/AISangam/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow)

This repo is not a fork because it aims to remove the large data (movies and jpeg) files included in [AISangam's repo](https://github.com/AISangam/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow).

Instructions can be found [here](http://www.aisangam.com/blog/real-time-face-recognition-using-facenet/)

# Known issues

* The code provides incorrect results if only one image per person is used during training
* The code does not work correctly on images containing multiple people
