import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

# from Edge import Edge, EdgeType
# from PuzzlePiece import PuzzlePiece

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

class PuzzlePiece:
    
    def __init__(self, segment_size):
        self.segment = segment_size
        self.piece_dim = 6*self.segment
        self.tab_shift = self.segment//8
        
    def create_mask_base(self):
        #Creates a square in the middle of a 6x6 segment square
        self.piece_mask = np.zeros((self.piece_dim, self.piece_dim, 4), dtype=np.uint8)
        base_end = self.piece_dim - self.segment
        self.piece_mask[self.segment: base_end, self.segment: base_end] = 255

    
    def add_tab(self, edge):
        match edge.edge_type:
            case EdgeType.TOP:
                #print("tab on top")
                self.piece_mask[:self.segment] = np.bitwise_or(self.piece_mask[:self.segment], edge.data)
            case EdgeType.LEFT:
                #print("tab on left")
                self.piece_mask[:,:self.segment] = np.bitwise_or(self.piece_mask[:,:self.segment], edge.data)
            case EdgeType.BTM:
                #print("tab on bottom")
                #print(self.piece_mask.shape)
                self.piece_mask[-self.segment:] = np.bitwise_or(self.piece_mask[-self.segment:], edge.data)
            case EdgeType.RIGHT:
                #print("tab on right")
                self.piece_mask[:,-self.segment:] = np.bitwise_or(self.piece_mask[:,-self.segment:], edge.data)
            case _:
                print("Cannot add tab with provided edge.")

    def add_space(self, edge):
        temp_mask = cv2.bitwise_not(edge.data)[:,:,3]
        match edge.edge_type:
            #type of edge means represents tab placement, a space would be placed in the opposite side
            case EdgeType.TOP:
                #print("space in bottom")
                #print(temp_mask.shape,self.piece_mask[-2*self.segment:-self.segment].shape )
                self.piece_mask[-2*self.segment:-self.segment] = cv2.bitwise_and(self.piece_mask[-2*self.segment:-self.segment], 
                                                                                 self.piece_mask[-2*self.segment:-self.segment], 
                                                                                 mask=temp_mask)
            case EdgeType.LEFT:
                #print("space in right")
                self.piece_mask[:,-2*self.segment:-self.segment] = cv2.bitwise_and(self.piece_mask[:,-2*self.segment:-self.segment],
                                                                                   self.piece_mask[:,-2*self.segment:-self.segment],
                                                                                   mask=temp_mask)
            case EdgeType.BTM:
                #print("space in top")
                #print(self.piece_mask.shape)
                self.piece_mask[self.segment:2*self.segment] = cv2.bitwise_and(self.piece_mask[self.segment:2*self.segment],
                                                                               self.piece_mask[self.segment:2*self.segment],
                                                                               mask=temp_mask)
            case EdgeType.RIGHT:
                #print("space in left")
                self.piece_mask[:,self.segment:2*self.segment] = cv2.bitwise_and(self.piece_mask[:,self.segment:2*self.segment],
                                                                                 self.piece_mask[:,self.segment:2*self.segment],
                                                                                 mask=temp_mask)
            case _:
                print("Cannot add tab with provided edge.")


class PuzzleGenerator:
    def __init__(self, img_path, rows, cols):
        #forces the img to be a squared img assuming h < w
        self.rows = rows
        self.cols = cols
        self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2BGRA)
        h,w,_ = self.img.shape
        self.img = self.img[:,:h]
        h,w,_ = self.img.shape

        #creates directory for this puzzle if it does not exist
        self.directory = f"{img_path[:-4]}/{rows}_x_{cols}_puzzle"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        #padding with border of width segment allows us to deal with border pieces easily
        self.segment = h // (4*rows)
        self.img = np.pad(self.img, ((self.segment, self.segment),(self.segment, self.segment), (0,0)))

        self.edges_arr = np.empty((rows,cols-1,2), dtype=object)
        self.pieces_arr = np.empty((rows,cols), dtype=object)

    def generate(self):
        #generates row x col-1 x 2 matrix of edges since thats the number of inner edges
        for i in range(self.rows):
            for j in range(self.cols-1):
                for k in range(2):
                    self.edges_arr[i,j,k] = Edge(self.segment)
                    if k == 0:
                    #randomizes the type of tab that will be created for this edge    
                        #in first dim we store left/right edges
                        self.edges_arr[i,j,k].create_tab(random.choice([EdgeType.LEFT,EdgeType.RIGHT]))
                    else:
                        #in second dim we store top/bottom edges
                        self.edges_arr[i,j,k].create_tab(random.choice([EdgeType.TOP,EdgeType.BTM]))
        
        for i in range(self.rows):
            for j in range(self.cols):
                temp_piece = PuzzlePiece(self.segment)
                temp_piece.create_mask_base()
        
                #print(i,j)
                if i != 0:
                    #add tab or space at the top'
                    #print(i,j)
                    if self.edges_arr[j,i-1,1].edge_type == EdgeType.TOP:
                        temp_piece.add_tab(self.edges_arr[j,i-1,1])
                    elif self.edges_arr[j,i-1,1].edge_type == EdgeType.BTM:
                        temp_piece.add_space(self.edges_arr[j,i-1,1])
                    else:
                        print("Accessing wrong set of edges left/right")
                if j != 0:
                    #add tab or space to the left
                    #print(i,j)
                    if self.edges_arr[i,j-1,0].edge_type == EdgeType.LEFT:
                        temp_piece.add_tab(self.edges_arr[i,j-1,0])
                    elif self.edges_arr[i,j-1,0].edge_type == EdgeType.RIGHT:
                        temp_piece.add_space(self.edges_arr[i,j-1,0])
                    else:
                        print("Accessing wrong set of edges top/bottom")
                if i != self.rows-1:
                    #add tab or space at the bottom
                    #print(edges_arr.shape)
                    #print(i,j)
                    if self.edges_arr[j,i,1].edge_type == EdgeType.TOP:
                        temp_piece.add_space(self.edges_arr[j,i,1])
                    elif self.edges_arr[j,i,1].edge_type == EdgeType.BTM:
                        temp_piece.add_tab(self.edges_arr[j,i,1])
                    else:
                        print("Accessing wrong set of edges left/right")
                if j != self.cols-1:
                    #add tab or space to the right
                    #print(i,j)
                    if self.edges_arr[i,j,0].edge_type == EdgeType.LEFT:
                        temp_piece.add_space(self.edges_arr[i,j,0])
                    elif self.edges_arr[i,j,0].edge_type == EdgeType.RIGHT:
                        temp_piece.add_tab(self.edges_arr[i,j,0])
                    else:
                        print("Accessing wrong set of edges top/bottom")
                        
                # self.pieces_arr[i,j] = temp_piece
                
                mask_out = cv2.subtract(temp_piece.piece_mask, 
                                        self.img[i*temp_piece.piece_dim-(i*2*self.segment):
                                                 (i+1)*temp_piece.piece_dim-(i*2*self.segment), 
                                                 j*temp_piece.piece_dim-(j*2*self.segment):
                                                 (j+1)*temp_piece.piece_dim-(j*2*self.segment)])
                self.pieces_arr[i,j] = cv2.subtract(temp_piece.piece_mask, mask_out)
        
                cv2.imwrite(f'{self.directory}/piece_{i}_{j}.png', self.pieces_arr[i,j])