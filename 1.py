import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#glove = pd.read_csv('C:/Users/MH/Desktop/MyCode/NLP/glove.840B.300d/go.txt', sep=' ', header = None, index_col = 0)
#print(glove)
#print(glove[','])
#print(train_data['Phrase'][0:3])
#a = train_data['Phrase'][0]
#print(a)
#b = glove['the']
#word = glove.loc['the'].tolist()
#a = torch.Tensor(word)
#print(a.size(),a)
a = 'C:/Users/MH/Desktop/MyCode/NLP/glove.840B.300d'
b = '/go.txt'
c = a+b
print(c)
