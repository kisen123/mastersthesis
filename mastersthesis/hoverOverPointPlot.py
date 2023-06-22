# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:15:54 2023

@author: kisen
"""


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)
import os

imagePath = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts/syntheticFolder/images"
img_paths = [os.path.join(imagePath, "0-0.tif").replace("\\", "/"),
          os.path.join(imagePath, "0-1.tif").replace("\\", "/"),
          os.path.join(imagePath, "0-2.tif").replace("\\", "/")]

# Generate data x, y for scatter and an array of images.
x = np.arange(20)
y = np.random.rand(len(x))
arr = np.empty((len(x),10,10))

def createFig():
    
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(x,y, ls="", marker="o")
    
    return fig, ax, line


def annotationBox(imagesTensor):
    # create the annotations box
    im = OffsetImage(imagesTensor[0,:,:], zoom=5)
    xybox=(50., 50.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    
    return xybox, ab, im



# CODE FROM https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
def hover(event, imagesTensor):
    
    xybox, ab, im = annotationBox()
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(imagesTensor[ind,:,:])
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)           
plt.show()



#%%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# Generate data x, y for scatter and an array of images.
size_dataset = 20
x = np.arange(size_dataset)
y = np.random.rand(len(x))
arr = np.random.rand(size_dataset, 28, 28)

# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x, y, ls="", marker="o")

# create the annotations box
img_height, img_width = arr.shape[1], arr.shape[2]
zoom = 1.5  # Adjust this zoom level to change the image size in the plot
im = OffsetImage(arr[0,:,:], zoom=zoom)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(arr[ind,:,:])
        
        im.set_zoom(zoom*(1/img_height))  # Adjust zoom level to match image dimensions
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)           
plt.show()



#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# Generate data x, y for scatter and an array of images.
size_dataset = 20
x = np.arange(size_dataset)
y = np.random.rand(len(x))
arr = np.random.rand(size_dataset, 28, 28)

# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x, y, ls="", marker="o")

# create the annotations box
img_height, img_width = arr.shape[1], arr.shape[2]
zoom = 1.5  # Adjust this zoom level to change the image size in the plot
im = OffsetImage(arr[0,:,:], zoom=zoom)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(arr[ind,:,:])
        
        im.set_zoom(zoom)  # Adjust zoom level to match image dimensions
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)           

plt.show()