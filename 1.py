import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

#glove = pd.read_csv('G:/MHwork/MHdata/NLP/glove.840B.300d/glove.840B.300d.txt', sep='\ ', header = None, index_col = 0, nrows=100,engine='python')
#print(glove)
#print(glove[','])
#print(train_data['Phrase'][0:3])
#a = train_data['Phrase'][0]
#print(a)
#b = glove['the']
#word = glove.loc['the'].tolist()
#a = torch.Tensor(word)
#print(a.size(),a)
#train_data = pd.read_csv('C:/Users/MH/Desktop/MyCode/NLP/Sentiment-Analysis-on-Movie-Reviews/data/train.tsv',sep='\t', nrows=5)
#print(train_data)
#a = train_data['Phrase']
#print('len_a:',len(a))

def f(isa = True):
    a = 3
    if isa:
        a = a+1
    print(a)

#f(isa = False)
"""
time_start = time.time()
voca = torch.load('G:/MHwork/MHdata/NLP/glove/voca.100d.pth')
time_mid = time.time()
print('load time cost:%fs'%(time_mid-time_start))
A = 'a'
if A in voca:
    print(A,  'in voca', voca.index(A))
time_end = time.time()
print('search time cost:%fs'%(time_end-time_mid))
print(voca[-1])
#print(glove)
#print(glove.loc[A])

a = 7.235
print('a取整是：%d'%a)

path = 'G:\\MHwork\\MHdata\\NLP\\Sentiment-Analysis-on-Movie-Reviews\\embdata\\train_data_tensor.pth'
a = torch.randn(3,5)
print(a)
b = a.max(1)[1]
print(b.size(0))
path = 'G:\\MHwork\\MHdata\\NLP\\Sentiment-Analysis-on-Movie-Reviews\\embdata\\temp.csv'
result = []
temp={}
for i in range(4):
    temp={}
    temp['PhraseId'] = i
    temp['Sentiment'] = 2
    result.append(temp)
df = pd.DataFrame(result, columns=['PhraseId','Sentiment'])
print(df)
df.to_csv(path, index = False)
a = pd.read_csv(path)
print(a)
"""
"""
input = torch.randn(2, 3, 4)
out = torch.cat((input[0,:,:],input[1,:,:]), dim = 1)
print(input,out)

a = nn.Linear(5,2)
print(a.weight,a.bias)

A = [1,2,3,4]
B = [4,5,6,7]
a = np.array([[1,2,3,5],[1,2,3,3]])
print(a,a.argmax(0))
"""

a = torch.randn(3,2)
b = torch.randn(3,2)
print(a, 10*a)




