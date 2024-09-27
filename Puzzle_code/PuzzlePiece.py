import cv2
import numpy as np
from Edge import Edge, EdgeType


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
        