import os
mypath = os.path.dirname(__file__)

from packages.preprocess import preprocesses
input_datadir = mypath + '/train_images'
output_datadir = mypath + '/pre_images'

obj=preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


