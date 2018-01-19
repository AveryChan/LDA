# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from gensim import corpora, models
import pandas as pd
import csv

training_set = []
with open('H:\\MineCos\\Data\\Ing Fact_SS.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_set.append((row['\xef\xbb\xbfIngredient'],row['Functions']) )
dictionary = corpora.Dictionary(training_set)
dictionary.save('H:\\MineCos\\Dictionary/Training_Ing Facts.dict')



def train_lda_model_gensim(corpus, dictionary, total_topics=10):
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]
    Lda = models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=total_topics, 
                   iterations=1000, id2word = dictionary, passes=50)
    ldamodel.save('H:\\MineCos\\TopicModels_func/model_%02dt.lda' % total_topics)
    return ldamodel

#Training
for i in range(50,90):
    print('traning %s-topic model' % (i+1))
    train_lda_model_gensim(training_set,dictionary,i+1)


##############################################################
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from scipy.spatial.distance import cosine

testing_set = []
with open('H:\MineCos\Data\Sunscreen with product type.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        testing_set.append(( row['receipe 2'],row['receipe 3'],
                             row['receipe 4'],row['receipe 5'], row['receipe 6'],
                             row['receipe 7'], row['receipe 8'],row['receipe 9'])) 
dictionary = corpora.Dictionary(testing_set)
dictionary.save('H:\\MineCos\\Dictionary/Testing_8ing.dict')

#Evaluating
product_id =[]
product_id=np.loadtxt('H:\MineCos\Data\Sunscreen with product type.csv',dtype=str,
                     delimiter=',',skiprows=1,usecols=(0))


for i in range(50,90):
    doc_ldas = []
    lda = models.ldamodel.LdaModel.load('H:\\MineCos/Topic Models\TopicModels_func/model_%02dt.lda'% (i+1))
    for doc in testing_set:   
        temp = [0] * (i + 1)
        for (key, value) in lda[dictionary.doc2bow(doc)]:
            temp[key]= value
        doc_ldas.append(temp)
    doc_ldas=np.array(doc_ldas)
    
    SimMat_cos = 1-pairwise_distances(doc_ldas, metric="cosine")
    
    for column in range(399):
        for row in range( len(SimMat_cos)):
                if SimMat_cos[row,column] < 0.2:
                    SimMat_cos[row,column] = 5
                elif SimMat_cos[row,column] <0.4:
                    SimMat_cos[row,column] = 4
                elif SimMat_cos[row,column] <0.6:
                    SimMat_cos[row,column] = 3
                elif SimMat_cos[row,column] <0.8:
                    SimMat_cos[row,column] = 2
                else:
                    SimMat_cos[row,column] = 1

    print('Save %s-topic Similarity Matrix' % (i+1))

    #write into dataframe
    SimMat_cos = pd.DataFrame(data=SimMat_cos[0:35,0:],    # value
                        index = ['Product '+ product_id[k] for k in range(35)],  
                        columns=['Product '+product_id[k] for k in range( len(SimMat_cos))] ) 
    
    #save results into excel
    writer = pd.ExcelWriter('H:\MineCos\Similarity Matrix\%02dT_cos.xlsx' % (i+1))
    SimMat_cos.to_excel(writer,'Sheet1')
    writer.save()
    print('SimMat %02d is saved')% (i+1)





