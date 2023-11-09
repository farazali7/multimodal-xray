'''
TRAINING SCRIPT
'''

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cross_attention import CrossAttentionBlock

class TransformerGenModel:
    def __init__(self, learning_rate, num_iter):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #TODO: train_dataset = # the training data
        #TODO: train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        self.learning_rate = learning_rate
        self.num_iter = num_iter

        #TODO: initialize other model components: encoder, bert, decoder
        self.encoder = Encoder().to(device)
        self.cross_attn = CrossAttentionBlock().to(device)
        self.bert = BertModel().to(device)
        self.decoder = Decoder().to(device)

        # TODO: Define the loss function - Ask Faraz if MSE loss is good to compare 2 imgs
        self.mse_loss = nn.MSELoss()

        # Optimizer
        params = list(encoder.parameters()) + list(cross_attn.parameters()) + list(bert.parameters()) + list(decoder.parameters())
        self.optimizer = optim.Adam(params, lr=0.001)


    def train(self):
        loss_array = []

        for i in range(self.num_iter):
            for i, (input_data, target_data) in enumerate(train_loader):
                # Move data to the appropriate device
                input_data = input_data.to(device)
                target_data = target_data.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass through the encoder
                encoder_output = self.encoder(input_data)

                 # Pass the text to be tokenizd to BERT
                bert_output = self.bert(text_input)

                # Pass the encoder output and input_data to cross-attention
                # change cross attn input params
                cross_attn_output = self.cross_attn(encoder_output, bert_output)

               

                # Decode the output of x attn
                decoded_output = self.decoder(cross_attn_output)

                # Calculate the MSE loss
                loss = self.mse_loss(decoded_output, target_data)

                loss_array.append(loss)

                # Backpropagation
                loss.backward()

                # Update the weights
                self.optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        # Plot the loss
        self.plot_loss(loss_array)

    def plot_loss(self, loss_array):
        plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
        plt.title('Plot of the Loss function')
        plt.xlabel('epochs')
        plt.ylabel('Train Loss')
        plt.show()
