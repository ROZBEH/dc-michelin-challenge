import cPickle
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from unidecode import unidecode
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation


def string_change(string):
    string = unicode(string, "utf-8")
    string = unidecode(string)
    word1 = " ".join(re.findall("[a-zA-Z]+", string))
    word1 = word1.lower()
    word1 = word1.split()
    word1 = list(set(word1))
    return word1

Featues_SF_no_Mich = cPickle.load(open('Featues_SF_no_Mich.p', 'rb'))
label_SF_no_Mich = cPickle.load(open('label_SF_no_Mich.p', 'rb'))

Features_SF_Mich = cPickle.load(open('Features_SF_Mich.p', 'rb'))
label_SF_Mich = cPickle.load(open('label_SF_Mich.p', 'rb'))

Features_NY = cPickle.load(open('Features_NY.p', 'rb'))
label_NY = cPickle.load(open('label_NY.p', 'rb'))

Features_DC = cPickle.load(open('Features_DC.p', 'rb'))

Train_Features = Featues_SF_no_Mich + Features_SF_Mich+Features_NY
Train_Label = label_SF_no_Mich + label_SF_Mich+label_NY

Test_Features = Features_DC



def feature_transition_Test(Train_Features, uniq_text):
    All_text = []
    for ii,feat in enumerate(Train_Features):
        temp = []
        for item in feat[-1]:
            temp = temp + string_change(item)
        temp = list(set(temp))
        Train_Features[ii][-1] = temp
        All_text += temp




    for ii,feat in enumerate(Train_Features):
        temp = []
        for item in feat[-1]:
            if item in uniq_text:
                temp.append(uniq_text.index(item))
        QQ = [0]*len(uniq_text)
        for item in temp:
            QQ[item] = 1
        Train_Features[ii][-1] = QQ

    for ii,feat in enumerate(Train_Features):
        if feat[3][0] == False:
            Train_Features[ii][3][0] = 0
        else:
            Train_Features[ii][3][0] = 1
            
        if feat[4][0] == False:
            Train_Features[ii][4][0] = 0
        else:
            Train_Features[ii][4][0] = 1
            
    for ii,feat in enumerate(Train_Features):
        if len(feat) > 6:
            del Train_Features[ii][-2]
    return Train_Features


# working with the text kind of feature
def feature_transition(Train_Features):
    All_text = []
    for ii,feat in enumerate(Train_Features):
        temp = []
        for item in feat[-1]:
            temp = temp + string_change(item)
        temp = list(set(temp))
        Train_Features[ii][-1] = temp
        All_text += temp


    wynik = {}
    for i in All_text:
        if i in wynik:
             wynik[i] += 1
        else:
             wynik[i] = 1

    # just considering the unique text
    init_text = list(set(All_text))

    uniq_text = []
    for item in init_text:
        if wynik[item] > 8:
            uniq_text.append(item)

    for ii,feat in enumerate(Train_Features):
        temp = []
        for item in feat[-1]:
            if item in uniq_text:
                temp.append(uniq_text.index(item))
        QQ = [0]*len(uniq_text)
        for item in temp:
            QQ[item] = 1
        Train_Features[ii][-1] = QQ

    for ii,feat in enumerate(Train_Features):
        if feat[3][0] == False:
            Train_Features[ii][3][0] = 0
        else:
            Train_Features[ii][3][0] = 1
            
        if feat[4][0] == False:
            Train_Features[ii][4][0] = 0
        else:
            Train_Features[ii][4][0] = 1
            
    for ii,feat in enumerate(Train_Features):
        if len(feat) > 6:
            del Train_Features[ii][-2]
    return Train_Features,uniq_text

Train_Features,uniq_text = feature_transition(Train_Features)
Test_Features = feature_transition_Test(Test_Features,uniq_text)

Train_Feat = []
Test_Feat = []




for lis in Train_Features:
    Train_Feat.append([item for sublist in lis for item in sublist])

for lis in Test_Features:
    Test_Feat.append([item for sublist in lis for item in sublist])


    

y_label_train = np.asarray(Train_Label)

def norm_feat(Train_Feat):
    star = []
    num_rev = []
    start_1 = []
    start_2 = []
    end_1 = []
    end_2 = []
    import math
    for ii, ba in enumerate(Train_Feat):
        star.append(ba[1])
        num_rev.append(int(round(ba[2]/float(50))))
        start_1.append(ba[4])
        end_1.append(ba[5])
        start_2.append(ba[7])
        end_2.append(ba[8])
        
    star = list(set(star))
    num_rev = list(set(num_rev))
    start_1 = list(set(start_1))
    start_2 = list(set(start_2))
    end_1 = list(set(end_1))
    end_2 = list(set(end_2))

    for ii, ba in enumerate(Train_Feat):
        
        hey = [0]*len(star)
        hey [star.index(ba[1])] = 1
        Train_Feat[ii][1] =hey

        hey = [0]*len(num_rev)
        hey [num_rev.index(int(round(ba[2]/float(50))))] = 1
        Train_Feat[ii][2] = hey

        hey = [0]*len(start_1)
        hey [start_1.index(ba[4])] = 1
        Train_Feat[ii][4] = hey

        hey = [0]*len(end_1)
        hey [end_1.index(ba[5])] = 1
        Train_Feat[ii][5] = hey

        hey = [0]*len(start_2)
        hey [start_2.index(ba[7])] = 1
        Train_Feat[ii][7] = hey

        hey = [0]*len(end_2)
        hey [end_2.index(ba[8])] = 1
        Train_Feat[ii][8] = hey
    return Train_Feat

######################################
##Train_Feat_1 = norm_feat(Train_Feat)
##Test_Feat_1 = norm_feat(Test_Feat)
##Train_Feat = []
##Test_Feat = []
##
##for I,it in enumerate(Train_Feat_1):
##    temp =[]
##    for item in it:
##        if type(item) == int:
##            temp.append(item)
##        else:
##            temp+= item
##    Train_Feat.append(temp)
##
##for I,it in enumerate(Test_Feat_1):
##    temp =[]
##    for item in it:
##        if type(item) == int:
##            temp.append(item)
##        else:
##            temp+= item
##    Test_Feat.append(temp)
########################################
X_train = np.asarray(Train_Feat)
X_test = np.asarray(Test_Feat)

logistic = LogisticRegression()
classify = logistic.fit(X_train,y_label_train)

Y_predict = logistic.predict(X_test)

last_file = open("DC_Predict.txt",'w')
for i in range(Y_predict.shape[0]):
    last_file.write(str(Y_predict[i]))
    last_file.write('\n')
last_file.close()

