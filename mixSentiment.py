import pandas as pd
import numpy as np 
from GloVeEmb import root_path

test_sentiment_path1 = root_path + '/test_sentiment_V1.csv'
test_sentiment_path2 = root_path + '/test_sentiment_V2.csv'
test_sentiment_path3 = root_path + '/test_sentiment_V3.csv'
test_sentiment_path4 = root_path + '/test_sentiment_4.csv'
test_sentiment_path5 = root_path + '/test_sentiment_5.csv'

mix_sentiment_path = root_path + '/mix3_sentiment.csv'
test_sentiment_V1 = pd.read_csv(test_sentiment_path1)
test_sentiment_V2 = pd.read_csv(test_sentiment_path2)
test_sentiment_V3 = pd.read_csv(test_sentiment_path3)
count = 0
num = len(test_sentiment_V1['Sentiment'])
S1 = test_sentiment_V1['Sentiment'].tolist()
#S2 = test_sentiment_V2['Sentiment'].tolist()
#S3 = test_sentiment_V3['Sentiment'].tolist()
S4 = test_sentiment_V2['Sentiment'].tolist()
S5 = test_sentiment_V3['Sentiment'].tolist()
#S = [S1,S4,S5]
count = 0
#mixS = []
#print(S3[0:5])
"""
for i in range(num):
    for j in range(5):
        if S[j][i] == 0:
            count[i,0] += 1
        elif S[j][i] == 1:
            count[i,1] += 1
        elif S[j][i] == 2:
            count[i,2] += 1
        elif S[j][i] == 3:
            count[i,3] += 1
        elif S[j][i] == 4:
            count[i,4] += 1
mixS = count.argmax(1)
"""
mixS = []
for i in range(num):
    if S1[i] == S4[i] and S1[i] == S5[i]:
        count += 1
        mixS.append(S1[i])
    elif S1[i] == S5[i]:
        mixS.append(S1[i])
    elif S5[i] == S4[i]:
        mixS.append(S5[i])
    elif S1[i] == S4[i]:
        mixS.append(S1[i])
    else:
        mixS.append(S5[i])


        
test_PhraseId = test_sentiment_V1['PhraseId']
result = []
for i in range(len(test_PhraseId)):
        temp={}
        temp['PhraseId'] = test_PhraseId[i]
        temp['Sentiment'] = mixS[i]
        result.append(temp)
df = pd.DataFrame(result, columns=['PhraseId','Sentiment'])
df.to_csv(mix_sentiment_path, index = False)

#print(count[0:20,:])




print('Acc is :', count/num)

