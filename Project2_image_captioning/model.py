import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob=0):
        super(DecoderRNN, self).__init__()
        
        self.caption_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
                            
        # define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        ''' Forward pass through the network '''
        
        # remove end token from captions
        captions = captions[:,:-1]
        
        # embed captions
        caption_embeds = self.caption_embeddings(captions)
        
        # concatenate the feature and caption embeds
        inputs = torch.cat((features.unsqueeze(1),caption_embeds),1)
        
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        out, hidden = self.lstm(inputs)
        
        # pass out through a droupout layer
        out = self.dropout(out)
                                
        # put out through the fully-connected layer
        out = self.fc(out)

        return out
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.fc(out.squeeze(1))
            _, predicted = out.max(1) 
            tokens.append(predicted.item())
            inputs = self.caption_embeddings(predicted) 
            inputs = inputs.unsqueeze(1)
        return tokens