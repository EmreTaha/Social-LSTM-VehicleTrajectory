'''
Train script for the Social LSTM model

Author: Anirudh Vemula
Date: 13th June 2017
'''
import torch
from torch.autograd import Variable

import argparse
import os
import time
import pickle

from model import SocialLSTM
from utils_vehicle import DataLoader
from grid import getSequenceGridMask
from st_graph import ST_GRAPH
from criterion import Gaussian2DLikelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=8,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')

    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')
    args = parser.parse_args()
    train(args)


def train(args):
    # datasets = [i for i in range(5)]
    datasets = [0, 1, 2]
    # Remove the leave out dataset from the datasets
    datasets.remove(args.leaveDataset)

    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length+1, datasets, forcePreProcess=True)

    # Construct the ST-graph object
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + 1)

    # Log directory
    log_directory = 'log/'
    log_directory += str(args.leaveDataset) + '/'

    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = 'save/'
    save_directory += str(args.leaveDataset) + '/'

    # Dump the arguments into the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'social_lstm_model_'+str(x)+'.tar')

    # Initialize net
    net = SocialLSTM(args)
    print(net)
    if args.use_cuda:
        net = net.cuda()

    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    learning_rate = args.learning_rate

    print('Training begin')
    best_val_loss = 100
    best_epoch = 0

    # Training
    for epoch in range(args.num_epochs):
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, _, d = dataloader.next_batch()

            # Construct the stgraph
            stgraph.readGraph(x)

            loss_batch = 0

            # For each sequence
            for sequence in range(dataloader.batch_size):
                # Get the data corresponding to the current sequence
                x_seq, d_seq = x[sequence], d[sequence]

                # Dataset dimensions
                if d_seq == 0 and datasets[0] == 0:
                    dataset_data = [640, 480]
                else:
                    dataset_data = [720, 576]

                # Compute grid masks
                grid_seq = getSequenceGridMask(x_seq, dataset_data, args.neighborhood_size, args.grid_size, args.use_cuda)

                # Get the node features and nodes present from stgraph
                nodes, _, nodesPresent, _ = stgraph.getSequence(sequence)

                # Construct variables
                nodes = Variable(torch.from_numpy(nodes).float())
                if args.use_cuda:                    
                    nodes = nodes.cuda()
                numNodes = nodes.size()[1]
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    hidden_states = hidden_states.cuda()
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    cell_states = cell_states.cuda()

                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _ = net(nodes[:-1], grid_seq[:-1], nodesPresent[:-1], hidden_states, cell_states)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.item()

                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            # Reset stgraph
            stgraph.reset()
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        loss_epoch /= dataloader.num_batches
        # Log loss values
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.valid_num_batches):
            # Get batch data
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

            # Read the st graph from data
            stgraph.readGraph(x)

            # Loss for this batch
            loss_batch = 0

            # For each sequence
            for sequence in range(dataloader.batch_size):
                # Get data corresponding to the current sequence
                x_seq, d_seq = x[sequence], d[sequence]

                # Dataset dimensions
                if d_seq == 0 and datasets[0] == 0:
                    dataset_data = [640, 480]
                else:
                    dataset_data = [720, 576]

                # Compute grid masks
                grid_seq = getSequenceGridMask(x_seq, dataset_data, args.neighborhood_size, args.grid_size, args.use_cuda)

                # Get node features and nodes present from stgraph
                nodes, _, nodesPresent, _ = stgraph.getSequence(sequence)


                # Construct variables
                nodes = Variable(torch.from_numpy(nodes).float())
                if args.use_cuda:                    
                    nodes = nodes.cuda()
                numNodes = nodes.size()[1]
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    hidden_states = hidden_states.cuda()
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    cell_states = cell_states.cuda()

                # Forward prop
                outputs, _, _ = net(nodes[:-1], grid_seq[:-1], nodesPresent[:-1], hidden_states, cell_states)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.item()

            # Reset the stgraph
            stgraph.reset()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        if dataloader.valid_num_batches != 0:            
            loss_epoch = loss_epoch / dataloader.valid_num_batches

            # Update best validation loss until now
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch = epoch

            print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
            print('Best epoch', best_epoch, 'Best validation loss', best_val_loss)
            log_file_curve.write(str(loss_epoch)+'\n')

        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    if dataloader.valid_num_batches != 0:        
        print('Best epoch', best_epoch, 'Best validation Loss', best_val_loss)
        # Log the best epoch and best validation loss
        log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()


if __name__ == '__main__':
    main()
