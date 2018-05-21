'''
Criterion for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 30th March 2017
'''


import torch
import numpy as np
from helper import getCoef


def Gaussian2DLikelihood(outputs, targets, nodesPresent, pred_length):
    '''
    Parameters:

    outputs: Torch variable containing tensor of shape seq_length x numNodes x 1 x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x 1 x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:

            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss


def Gaussian2DLikelihoodInference(outputs, targets, assumedNodesPresent, nodesPresent):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    '''
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(outputs.size()[0]):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:
            if nodeID not in assumedNodesPresent:
                continue
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss
