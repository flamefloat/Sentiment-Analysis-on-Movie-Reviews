import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from GloVeEmb import train_data_tensor_path, Phrase_Len_Set_path, Sentiment_Set_path, root_path, test_data_tensor_path

# model parameters
input_size = 100
hidden_size = 200
n_class = 5
epochs = 50
MODEL_PATH = root_path + '/model(n_hidden=200,numlayer=1, droup=0.5, 2BiLSTM, 2linear).pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAmodel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(SAmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, self.hidden_size, num_layers = self.num_layers, batch_first = True, bidirectional = True, dropout = 0.5)
        #self.encoder2 = nn.LSTM(hidden_size *2, hidden_size, batch_first = True, bidirectional = True)
        #self.encoder = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 50)
        self.fc2 = nn.Linear(50, n_class)


    def forward(self, embed_input_x, sentence_lens):
        x = nn.utils.rnn.pack_padded_sequence(embed_input_x, sentence_lens, batch_first = True, enforce_sorted = False)
        _, (h_n, c_n) = self.encoder(x)
        h_n = h_n.view(self.num_layers,2,h_n.size(1),self.hidden_size) #(num_layers, num_directions, batch, hidden_size)
        h_n = torch.cat((h_n[self.num_layers-1,0,:,:],h_n[self.num_layers-1,1,:,:]), dim = 1)

        h_n = self.dropout(h_n)
        output = self.fc1(h_n)
        output = self.dropout(F.relu(output))
        output = self.fc2(output)
        return output


def initNetParams(net):
    '''Init net parameters.'''
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            init.xavier_normal_(layer.weight)
            init.zeros_(layer.bias)
        """
        elif isinstance(layer, nn.BatchNorm2d):
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            init.normal_(layer.weight, std=1e-3)
            if layer.bias:
                init.constant_(layer.bias, 0)
        """
 


def train_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    time_start = time.time()
    for  epoch in range(epochs):
        for i, data in enumerate(train_loader):
            Phrase_Set, Phrase_Len_Set, Sentiment_Set = data
            Phrase_Set = Phrase_Set.to(DEVICE)
            Phrase_Len_Set = Phrase_Len_Set.to(DEVICE)
            Sentiment_Set = Sentiment_Set.to(DEVICE)
            output = model(Phrase_Set, Phrase_Len_Set)
            train_loss = criterion(output, Sentiment_Set) 
            if (i+1)%200 == 0:
                time_end = time.time()
                print('Epoch:', '%04d' % (epoch + 1), 'batch:', '%04d' % (i + 1),'loss =', '{:.6f}'.format(train_loss),'%d Phrases/s'%((200*128)/(time_end-time_start)))
                time_start = time.time()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    trained_model = {'net': model.state_dict()}
    print('end of training\n')
    torch.save(trained_model, MODEL_PATH)
    print('Trained neural network model has been saved\n')

if __name__ == "__main__":
    
    model = SAmodel(hidden_size, num_layers=1).to(DEVICE)
    model.train()
    initNetParams(model)
    print('loading train data...')
    load_time_start = time.time()
    Phrase_Set = torch.load(train_data_tensor_path)
    Phrase_Len_Set = torch.load(Phrase_Len_Set_path)
    Sentiment_Set = torch.load(Sentiment_Set_path)
    load_time_end = time.time()
    print('load data time cost:%ds'%(load_time_end-load_time_start))
    
    deal_dataset = TensorDataset(Phrase_Set, Phrase_Len_Set, Sentiment_Set)
    train_loader = DataLoader(deal_dataset, batch_size=256, shuffle=True)
    print('start trainging...')
    train_model(model, train_loader)
    """
    a = torch.randn(20,3)
    torch.save(a, MODEL_PATH)
    """

            



