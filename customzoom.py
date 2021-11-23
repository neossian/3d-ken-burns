#!/usr/bin/env python

import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './autozoom.mp4'
arguments_size = 1024
argumnets_fps = 30
arguments_zoom = 1.3
argument_poihorizontal = 0
argument_poivertical = 0

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
	if strOption == '--size' and strArgument != '': arguments_size = int(strArgument) #Size of the image
	if strOption == '--fps' and strArgument != '': arguments_fps = int(strArgument) #FrameRateOfOutput
	if strOption == '--zoom' and strArgument != '': arguments_zoom = float(strArgument) #Zoom amount
	if strOption == '--poih' and strArgument != '': argument_poihorizontal = float(strArgument) #Zoom amount
	if strOption == '--poiv' and strArgument != '': argument_poivertical = float(strArgument) #Zoom amount		
		
# end

##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = npyImage.shape[1]
	intHeight = npyImage.shape[0]

	fltRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(arguments_size * fltRatio), arguments_size)
	intHeight = min(int(arguments_size / fltRatio), arguments_size)

	npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	process_load(npyImage, {})

	objFrom = {
		'fltCenterU': intWidth / 2.0,
		'fltCenterV': intHeight / 2.0,
		'intCropWidth': int(math.floor(0.97 * intWidth)),
		'intCropHeight': int(math.floor(0.97 * intHeight))
	}
	
	targetWidth= int(round(objFrom.intCropWidth / arguments_zoom))
	targetHeight= int(round(objFrom.intCropHeight / arguments_zoom))
	targetCenterU = objFrom.fltCenterU + (argument_poihorizontal * objFrom.fltCenterU)
	targetCenterV = objFrom.fltCenterU + (argument_poivertical * objFrom.fltCenterV)
	if ((targetCenterU +(targetWidth/2))>intWidth): targetCenterU = intWidth - (targetWidth/2) - 2		
	if ((targetCenterU +targetWidth)<0): targetCenterU = targetWidth/2 + 2		
	if ((targetCenterV + targetHeight)<0):targetCenterV = targetHeight/2 + 2		
	if ((targetCenterV +(targetHeight/2))>intHeight): targetCenterV = intHeight - (targetHeight/2) - 2	
		
		
	objTo = {
		'fltCenterU': targetCenterU,
		'fltCenterV': targetCenterV ,
		'intCropWidth': targetWidth,
		'intCropHeight': targetHeight
	}	
	
	npyResult = process_kenburns({
		'fltSteps': numpy.linspace(0.0, 1.0, 120).tolist(),
		'objFrom': objFrom,
		'objTo': objTo,
		'boolInpaint': True
	})

	moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult ], fps=argumnets_fps).write_videofile(arguments_strOut)
# end
