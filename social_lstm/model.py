'''
Model script for the Social LSTM model

Author: Anirudh Vemula
Date: 13th June 2017
'''

import torch
import torch.nn as nn
from torch.autograd import Variable


class SocialLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialLSTM, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size

        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)
        if self.use_cuda:
            self.cell = self.cell.cuda()
        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        if self.use_cuda:
            self.input_embedding_layer = self.input_embedding_layer.cuda()
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)
        if self.use_cuda:
            self.tensor_embedding_layer = self.tensor_embedding_layer.cuda()        
        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        if self.use_cuda:
            self.output_layer = self.output_layer.cuda()  
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        if self.use_cuda:
            self.relu = self.relu.cuda()
            self.dropout =    self.dropout.cuda()

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]
        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor

    def forward(self, nodes, grids, nodesPresent, hidden_states, cell_states):
        '''
        Forward pass for the model
        params:
        nodes: Input positions
        grids: Grid masks
        nodesPresent: Peds present in each frame
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # Number of peds in the sequence
        numNodes = nodes.size()[1]

        # Construct the output variable
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum in range(self.seq_length):
            # Peds present in the current frame
            nodeIDs = nodesPresent[framenum]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = Variable(torch.LongTensor(nodeIDs))
            if self.use_cuda:
                list_of_nodes = list_of_nodes.cuda()
            if self.use_cuda:
                hidden_states = hidden_states.cuda()
            if self.use_cuda:
                cell_states = cell_states.cuda()
            if self.use_cuda:
                nodes = nodes.cuda()
            # Select the corresponding input positions
            nodes_current = torch.index_select(nodes[framenum], 0, list_of_nodes)

            # Get the corresponding grid masks
            grid_current = grids[framenum]
            if self.use_cuda:
                grid_current = grid_current.cuda()
            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, list_of_nodes)
            cell_states_current = torch.index_select(cell_states, 0, list_of_nodes)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))

            # Compute the output
            outputs[framenum*numNodes + list_of_nodes.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[list_of_nodes.data] = h_nodes
            cell_states[list_of_nodes.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
