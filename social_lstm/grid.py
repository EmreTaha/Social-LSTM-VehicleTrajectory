'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
import numpy as np
import torch
from torch.autograd import Variable


def getGridMask(frame, dimensions, neighborhood_size, grid_size):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''

    # Maximum number of pedestrians
    mnp = frame.shape[0]
    width, height = dimensions[0], dimensions[1]

    frame_mask = np.zeros((mnp, mnp, grid_size**2))

    width_bound, height_bound = (neighborhood_size/(width*1.0))*2, (neighborhood_size/(height*1.0))*2

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):

        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        # For all the other peds
        for otherpedindex in range(mnp):

            # If the other pedID is the same as current pedID
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 1], frame[otherpedindex, 2]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getGridMaskInference(frame, dimensions, neighborhood_size, grid_size):
    mnp = frame.shape[0]
    width, height = dimensions[0], dimensions[1]

    frame_mask = np.zeros((mnp, mnp, grid_size**2))

    width_bound, height_bound = (neighborhood_size/(width*1.0))*2, (neighborhood_size/(height*1.0))*2

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):
        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 0], frame[pedindex, 1]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        # For all the other peds
        for otherpedindex in range(mnp):
            # If the other pedID is the same as current pedID
            if otherpedindex == pedindex:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 0], frame[otherpedindex, 1]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue
            
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size, using_cuda):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        # sequence_mask[i, :, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], dimensions, neighborhood_size, grid_size)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask
