# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:22:03 2023

@author: kisen
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# This script plots scatter plot points, and if the user hovers over the
# points, the images show up.

def scatter_plot_with_hover(x, y, imagesTensor, customColorsPointwise):
    # Create figure and plot scatter
    fig, ax = plt.subplots(1,1)
    scatter = ax.scatter(x, y, c=customColorsPointwise, edgecolors="k")

    # Create the annotations box
    zoom = 1.5  # Adjust this zoom level to change the image size in the plot
    img = OffsetImage(imagesTensor[0, 0, :, :], zoom=zoom, cmap='gray')
    xybox=(50., 50.)
    ab = AnnotationBbox(img, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

    # Add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)


    def hover(event):
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        
        # find out the indices within the array from the event
        indices = scatter.contains(event)[1]["ind"]

        # if the mouse is over the scatter points
        if len(indices) > 0:
            for ind in indices:
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                ab.xybox = (xybox[0]*ws, xybox[1]*hs)
                # make annotation box visible
                ab.set_visible(True)
                # place it at the position of the hovered scatter point
                ab.xy =(x[ind], y[ind])
                # set the image corresponding to that point
                img.set_data(imagesTensor[ind, 0, :, :])
                
                img.set_zoom(zoom)  # Adjust zoom level to match image dimensions
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    # plt.show()  
    
    return fig, ax
