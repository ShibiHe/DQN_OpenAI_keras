__author__ = 'frankhe'
"""image anti-flickering test """

import image_preprocessing as IP
import cPickle
import numpy
file1 = open('images_test', mode='r')
images = cPickle.load(file1)
# IP.imshow(images[0], True)
# IP.imshow(images[1], True)
# IP.imshow(images[2], True)
# IP.imshow(images[3], True)

a = numpy.maximum(images[0], images[1])
b = numpy.maximum(images[2], images[3])
c = numpy.maximum(a, b)
""" Only adding even and odd frames is not enough (DQN paper) """
IP.imshow(a, True)
""" At least need three or four frames"""
IP.imshow(c, True)

