'''
Utils script for the structural RNN implementation
Handles processing the input and target data in batches and sequences

Author : Anirudh Vemula
Date : 15th March 2017
'''

import os
import pickle
import numpy as np
import random


class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0, 1, 2], forcePreProcess=False, infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        self.used_data_dirs =['../data_vehicles/0','../data_vehicles/1','../data_vehicles/2']
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.used_data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = '../data_vehicles'

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Validation arguments
        self.val_fraction = 0.2

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:
            # define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')
            file_path = os.path.join(directory, 'norm_pixels.csv')

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])

            skip = 10

            for ind, frame in enumerate(frameList):

                # NOTE CHANGE
                if ind % skip != 0:
                    # SKip every n frames
                    continue

                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y])

                if (ind > numFrames * self.val_fraction) or (self.infer):
                    # At inference time, no validation data
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index].append(np.array(pedsWithPos))
                else:
                    valid_frame_data[dataset_index].append(np.array(pedsWithPos))

            dataset_index += 1

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print('Training data from dataset', dataset, ':', len(all_frame_data))
            print('Validation data from dataset', dataset, ':', len(valid_frame_data))
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        print('Total number of training batches:', self.num_batches * 2)
        print('Total number of validation batches:', self.valid_num_batches)
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2
        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
