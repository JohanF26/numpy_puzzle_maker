import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import random


class EdgeType(Enum):
    '''
    Invalid means its the edge of the puzzle board
    top, left, btm, right: indicate where the tab is located, which dictates the shape of the puzzle piece
    '''
    # TOP_INVALID = -1
    # LEFT_INVALID = -2
    # BTM_INVALID = -3
    # RIGHT_INVALID = -4
    NOT_SET = 0
    TOP = 1
    LEFT = 2
    BTM = 3
    RIGHT = 4

class Edge:        
    #Edges are always composed of 6x1 segments or 1x6 segments
    
    def __init__(self, segment_size):
        self.edge_type = EdgeType.NOT_SET
        self.segment = segment_size
        self.tab_shift = self.segment//8
        self.edge_dim = 6*self.segment
        self.data = None

    def create_tab(self, edge_type=EdgeType.NOT_SET):
        '''
        num_points: How many anchor points to put on the circle to create deformation
        deform_angle: number between [-270:270] indicating the location of the deformation
        k: factor by which to shrink the radius to create indent
        '''
        def helper(num_points, deform_angle):
            angle_step_size = 360 // num_points
        
            #Generate 50 evenly spaced theta values between 0-2pi (circle)
            theta = np.linspace(0, 2*np.pi)
            theta1 = deform_angle * np.pi / 180 #angle of deformation
            alpha = angle_step_size * np.pi / 180 #distance between points
            th = theta - theta1
            t = np.abs(np.where(th < np.pi, th, th - 2*np.pi)) / alpha
            r = np.where(t > 2, 1, k + (2*t**2-9*t+11)*t**2*(1-k)/4)
            
            fig, ax = plt.subplots()

            #100 ppi, figsize is passed in inches and accepts floats
            fig_size = self.segment / 100
            #print(fig_size)
            fig = plt.figure(frameon=False, figsize=(fig_size, fig_size))
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            ax.plot(r * np.cos(theta), r * np.sin(theta), color='white')
            ax.axis('equal')
            plt.fill(r * np.cos(theta), r * np.sin(theta), color='white')
            fig.canvas.draw()
            
            buf = fig.canvas.buffer_rgba()
            ncols, nrows = fig.canvas.get_width_height()
            #close all plots once done with them to avoid wasting memory
            plt.close('all')
            
            self.data = np.frombuffer(buf, dtype=np.uint8).reshape((nrows, ncols, 4))
            #self.size_offset = self.segment-self.data.shape[0]
            #print(self.size_offset)
        
        if edge_type:
            self.edge_type = edge_type
            num_points = random.randint(7,13)   
            k = random.randint(60,80) / 100
            match self.edge_type:
                #Each edge type has restrictions to where to create the distortion to have more cohesive tabs
                case EdgeType.TOP:
                    deform_angle = random.randint(0,180)
                    helper(num_points, deform_angle)

                    #bottom_padding = self.segment-self.tab_shift
                    bottom_padding = 0
                    top_padding = self.tab_shift
                    random_padding = [2*self.segment,3*self.segment]
                    random.shuffle(random_padding)
                    right_padding, left_padding = random_padding

                    self.data = np.pad(self.data, ((top_padding,bottom_padding), 
                                           (left_padding,right_padding), 
                                           (0,0)), mode='constant')[:self.segment]
                    
                case EdgeType.LEFT:
                    deform_angle = random.randint(90,270)
                    helper(num_points, deform_angle)

                    #right_padding = self.segment-self.tab_shift
                    right_padding = 0
                    left_padding = self.tab_shift
                    random_padding = [2*self.segment,3*self.segment]
                    random.shuffle(random_padding)
                    top_padding, bottom_padding = random_padding

                    self.data = np.pad(self.data, ((top_padding,bottom_padding), 
                                           (left_padding,right_padding), 
                                           (0,0)), mode='constant')[:,:self.segment]
                    
                case EdgeType.BTM:
                    deform_angle = random.randint(-180,0)
                    helper(num_points, deform_angle)

                    #top_padding = self.segment-self.tab_shift
                    top_padding = 0
                    bottom_padding = self.tab_shift
                    random_padding = [2*self.segment,3*self.segment]
                    random.shuffle(random_padding)
                    right_padding, left_padding = random_padding

                    self.data = np.pad(self.data, ((top_padding,bottom_padding), 
                                           (left_padding,right_padding), 
                                           (0,0)), mode='constant')[self.tab_shift:]
                    
                case EdgeType.RIGHT:
                    deform_angle = random.randint(-90,90)
                    helper(num_points, deform_angle)

                    #left_padding = self.segment-self.tab_shift
                    left_padding = 0
                    right_padding = self.tab_shift
                    random_padding = [2*self.segment,3*self.segment]
                    random.shuffle(random_padding)
                    top_padding, bottom_padding = random_padding

                    
                    #print(f"Correction: {self.correction}")
                    self.data = np.pad(self.data, ((top_padding,bottom_padding), 
                                           (left_padding,right_padding), 
                                           (0,0)), mode='constant')[:,self.tab_shift:]

            
            
        