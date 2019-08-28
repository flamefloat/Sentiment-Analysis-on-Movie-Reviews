import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from GloVeEmb import train_data_tensor_path, Phrase_Len_Set_path, Sentiment_Set_path, root_path

# model parameters
input_size = 300
hidden_size = 256
n_class = 5
MODEL_PATH = root_path + '/model.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAmodel():
    def __init__(self):
        super(SAmodel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, n_class)

    def forward(self, embed_input_x, sentence_lens):
        x = nn.utils.rnn.pack_padded_sequence(embed_input_x, sentence_lens, batch_first=True)
        _, (h_n, c_n) = self.encoder(x)
        output = self.out(h_n)
        return output

def train_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for  epoch in range(5):
        for i, data in enumerate(train_loader):
            Phrase_Set, Phrase_Len_Set, Sentiment_Set = data
            Phrase_Set = Phrase_Set.to(DEVICE)
            Phrase_Len_Set = Phrase_Len_Set.to(DEVICE)
            Sentiment_Set = Sentiment_Set.to(DEVICE)
            output = model(Phrase_Set, Phrase_Len_Set)
            train_loss = criterion(output, Sentiment_Set) 
            if (i+1)%200 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'batch:', '%04d' % (i + 1),'cost =', '{:.6f}'.format(train_loss))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    trained_model = {'net': model.state_dict()}
    print('end of training\n')
    torch.save(trained_model, MODEL_PATH)
    print('Trained neural network model has been saved\n')

if __name__ == "__main__":
    model = SAmodel().to(DEVICE)
    print('loading train data...')
    Phrase_Set = torch.load(train_data_tensor_path)
    Phrase_Len_Set = torch.load(Phrase_Len_Set_path)
    Sentiment_Set = torch.load(Sentiment_Set_path)
    deal_dataset = TensorDataset(Phrase_Set, Phrase_Len_Set, Sentiment_Set)
    train_loader = DataLoader(deal_dataset, batch_size=128, shuffle=True)
    print('start trainging...')
    train_model(model, train_loader)
    

            



