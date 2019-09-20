import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.data import DataLoader, TensorDataset
from GloVeEmb import test_data_tensor_path, test_Phrase_Len_Set_path, root_path
from GloVeEmb import train_data_tensor_path, Sentiment_Set_path, Phrase_Len_Set_path
from trainModel import SAmodel, DEVICE, MODEL_PATH

test_sentiment_path = root_path + '/test_sentiment_V3.csv'

def getSentiment(model, test_Phrase_data, test_Phrase_Len_Set):
    test_Phrase_data = test_Phrase_data#.to(DEVICE)
    test_Phrase_Len_Set = test_Phrase_Len_Set#.to(DEVICE)
    model.eval()
    with torch.no_grad():
        sentiment = model(test_Phrase_data, test_Phrase_Len_Set)
        #sentiment = sentiment.max(1)[1]
    return sentiment

def getAcc(label, predict):
    length = label.size(0)
    count_error = 0
    for i in range(length):
        if label[i] == predict[i]:
            count_error += 1
    return count_error/length


if __name__ =='__main__':
    model_data = torch.load(MODEL_PATH)
    samodel = SAmodel()#.to(DEVICE)
    samodel.load_state_dict(model_data['net'])

    label_Sentiment_Set = torch.load(Sentiment_Set_path)
    test_data = pd.read_csv( root_path + '/data/test.tsv',sep='\t')
    test_PhraseId = test_data['PhraseId']

    train_Phrase_data = torch.load(train_data_tensor_path)
    train_Phrase_Len_Set = torch.load(Phrase_Len_Set_path)
    sentiment = getSentiment(samodel, train_Phrase_data, train_Phrase_Len_Set)
    Acc = getAcc(label_Sentiment_Set, sentiment)
    print('Predictive accuracy is', Acc) # Predictive accuracy is 0.8369(training set)
    #torch.save(sentiment, test_sentiment_path)
    #"""
    

    test_Phrase_data = torch.load(test_data_tensor_path)
    test_Phrase_Len_Set = torch.load(test_Phrase_Len_Set_path)
    test_sentiment = getSentiment(samodel, test_Phrase_data, test_Phrase_Len_Set)
    result = []
    for i in range(len(test_PhraseId)):
        temp={}
        temp['PhraseId'] = test_PhraseId[i]
        temp['Sentiment'] = test_sentiment[i].item()
        result.append(temp)
    df = pd.DataFrame(result, columns=['PhraseId','Sentiment'])
    df.to_csv(test_sentiment_path, index = False)
    #"""

