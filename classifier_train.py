from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os


from packages.classifier import training

datadir = './pre_images'
modeldir =  './model'
classifier_dir = './classifier';
try:
	os.mkdir(classifier_dir)
except OSError:
	print('Directory already exists')
classifier_filename = classifier_dir+'/classifier.pkl'
print ("Training Start")
obj=training(datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")
