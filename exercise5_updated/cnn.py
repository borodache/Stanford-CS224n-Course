#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size:int, word_embed_size:int, max_len:int=21, kernel_size:int=5, padding:int=1):
        """
        @param char_embed_size int: w_{char}
        @param word_embed_size int: w_{embed}, aka the # of filters for language modeling conv1d
        @param max_len int: maximum length of words
        @param kernel_size int: numer of kernels, default 5
        @param padding int: padding scheme, default 1
        """
        super(CNN, self).__init__()
        self.num_filters = word_embed_size
        self.conv1d = nn.Conv1d(in_channels=char_embed_size,
                                out_channels=word_embed_size,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=True)

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        """
        Highway connection, formula (5), (6), (7)
        @param X_reshaped (torch.Tensor): input from character embeddings, shape (batch_size, char_embed_size, max_len)
        @output X_conv_out (torch.Tensor): output of CNN module, shape (batch_szize, word_embd_size)
        """
        # X_reshaped: (batch_size, char_embed_size, max_len)
        X_conv = self.conv1d(X_reshaped)
        X_conv_out = torch.max(X_conv, dim=2)[0]

        return X_conv_out

    ### END YOUR CODE

