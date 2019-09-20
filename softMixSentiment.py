import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.data import DataLoader, TensorDataset
from GloVeEmb import test_data_tensor_path, test_Phrase_Len_Set_path, root_path
#from GloVeEmb import train_data_tensor_path, Sentiment_Set_path, Phrase_Len_Set_path
from trainModel import SAmodel, DEVICE
from testModel import getSentiment

model_path3 = root_path + '/model(n_hidden=150, droup=0.5, 2BiLSTM, 2linear, init3).pth' #n_hidden = 200
model_path2 = root_path + '/model(n_hidden=150, droup=0.5, 2BiLSTM, 2linear, init2).pth'
model_path1 = root_path + '/model(n_hidden=100, droup=0.5, 2BiLSTM, 2linear, init1).pth'
model_path4 = root_path + '/model(n_hidden=125, droup=0.5, 2BiLSTM, 2linear, init4).pth'
model_path5 = root_path + '/model(n_hidden=175, droup=0.5, 2BiLSTM, 2linear, init5).pth'
model_path6 = root_path + '/model(n_hidden=100,numlayer=3, droup=0.5, 2BiLSTM, 2linear).pth'
model_path7 = root_path + '/model(n_hidden=125,numlayer=3, droup=0.5, 2BiLSTM, 2linear).pth'
model_path8 = root_path + '/model(n_hidden=150,numlayer=3, droup=0.5, 2BiLSTM, 2linear).pth'
model_path9 = root_path + '/model(n_hidden=175,numlayer=3, droup=0.5, 2BiLSTM, 2linear).pth'
model_path10 = root_path + '/model(n_hidden=200,numlayer=3, droup=0.5, 2BiLSTM, 2linear).pth'
model_path11 = root_path + '/model(n_hidden=100,numlayer=1, droup=0.5, 2BiLSTM, 2linear).pth'
model_path12 = root_path + '/model(n_hidden=125,numlayer=1, droup=0.5, 2BiLSTM, 2linear).pth'
model_path13 = root_path + '/model(n_hidden=150,numlayer=1, droup=0.5, 2BiLSTM, 2linear).pth'
model_path14 = root_path + '/model(n_hidden=175,numlayer=1, droup=0.5, 2BiLSTM, 2linear).pth'
model_path15 = root_path + '/model(n_hidden=200,numlayer=1, droup=0.5, 2BiLSTM, 2linear).pth'

test_sentiment_path = root_path + '/softMix15_test_sentiment.csv'

model_data1 = torch.load(model_path1)
model_data2 = torch.load(model_path2)
model_data3 = torch.load(model_path3)
model_data4 = torch.load(model_path4)
model_data5 = torch.load(model_path5)
model_data6 = torch.load(model_path6)
model_data7 = torch.load(model_path7)
model_data8 = torch.load(model_path8)
model_data9 = torch.load(model_path9)
model_data10 = torch.load(model_path10)
model_data11 = torch.load(model_path11)
model_data12 = torch.load(model_path12)
model_data13 = torch.load(model_path13)
model_data14 = torch.load(model_path14)
model_data15 = torch.load(model_path15)

samodel1 = SAmodel(100,2)#.to(DEVICE)
samodel1.load_state_dict(model_data1['net'])
samodel2 = SAmodel(150,2)#.to(DEVICE)
samodel2.load_state_dict(model_data2['net'])
samodel3 = SAmodel(200,2)#.to(DEVICE)
samodel3.load_state_dict(model_data3['net'])
samodel4 = SAmodel(125,2)#.to(DEVICE)
samodel4.load_state_dict(model_data4['net'])
samodel5 = SAmodel(175,2)#.to(DEVICE)
samodel5.load_state_dict(model_data5['net'])
samodel6 = SAmodel(100,3)#.to(DEVICE)
samodel6.load_state_dict(model_data6['net'])
samodel7 = SAmodel(125,3)#.to(DEVICE)
samodel7.load_state_dict(model_data7['net'])
samodel8 = SAmodel(150,3)#.to(DEVICE)
samodel8.load_state_dict(model_data8['net'])
samodel9 = SAmodel(175,3)#.to(DEVICE)
samodel9.load_state_dict(model_data9['net'])
samodel10 = SAmodel(200,3)#.to(DEVICE)
samodel10.load_state_dict(model_data10['net'])
samodel11 = SAmodel(100,1)#.to(DEVICE)
samodel11.load_state_dict(model_data11['net'])
samodel12 = SAmodel(125,1)#.to(DEVICE)
samodel12.load_state_dict(model_data12['net'])
samodel13 = SAmodel(150,1)#.to(DEVICE)
samodel13.load_state_dict(model_data13['net'])
samodel14 = SAmodel(175,1)#.to(DEVICE)
samodel14.load_state_dict(model_data14['net'])
samodel15 = SAmodel(200,1)#.to(DEVICE)
samodel15.load_state_dict(model_data15['net'])

test_data = pd.read_csv( root_path + '/data/test.tsv',sep='\t')
test_PhraseId = test_data['PhraseId']

test_Phrase_data = torch.load(test_data_tensor_path)
test_Phrase_Len_Set = torch.load(test_Phrase_Len_Set_path)

test_sentiment1 = F.softmax(getSentiment(samodel1, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment2 = F.softmax(getSentiment(samodel2, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment3 = F.softmax(getSentiment(samodel3, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment4 = F.softmax(getSentiment(samodel4, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment5 = F.softmax(getSentiment(samodel5, test_Phrase_data, test_Phrase_Len_Set), dim=1)

test_sentiment6 = F.softmax(getSentiment(samodel6, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment7 = F.softmax(getSentiment(samodel7, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment8 = F.softmax(getSentiment(samodel8, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment9 = F.softmax(getSentiment(samodel9, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment10 = F.softmax(getSentiment(samodel10, test_Phrase_data, test_Phrase_Len_Set), dim=1)

test_sentiment11 = F.softmax(getSentiment(samodel11, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment12 = F.softmax(getSentiment(samodel12, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment13 = F.softmax(getSentiment(samodel13, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment14 = F.softmax(getSentiment(samodel14, test_Phrase_data, test_Phrase_Len_Set), dim=1)
test_sentiment15 = F.softmax(getSentiment(samodel15, test_Phrase_data, test_Phrase_Len_Set), dim=1)

test_sentiment = test_sentiment1 + test_sentiment2 + test_sentiment3 + test_sentiment4 + test_sentiment5 \
    + test_sentiment6 + test_sentiment7 + test_sentiment8 + test_sentiment9 + test_sentiment10 \
        + test_sentiment11 + test_sentiment12 + test_sentiment13 + test_sentiment14 + test_sentiment15

test_sentiment = test_sentiment.max(1)[1]

result = []
for i in range(len(test_PhraseId)):
    temp={}
    temp['PhraseId'] = test_PhraseId[i]
    temp['Sentiment'] = test_sentiment[i].item()
    result.append(temp)
df = pd.DataFrame(result, columns=['PhraseId','Sentiment'])
df.to_csv(test_sentiment_path, index = False)