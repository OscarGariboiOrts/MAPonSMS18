# -⁻- coding: UTF-8 -*-

import os
import sys
import pandas as pd
import sklearn
from sklearn.svm import SVC
import pickle
from scipy.stats import skew
import numpy as np
import re
import csv
from sklearn.svm import LinearSVC


def load_data(ruta):
	'''This function loads the txt files into a list of text for each user.
	Input: path to data to be loaded.
	Output: list of text for each user'''
	files= [f for f in os.listdir(ruta) if f.endswith('.txt')]
	files.sort()
	todo=[]
	for filed in files:
		with open(ruta+filed, encoding='cp1252') as f:
			variable=f.read().replace('\n\n', ' ').replace('\n','')
			todo.append(variable)
	return todo

def load_label(ruta):
	'''This function loads the Truth file into a pandas dataframe
	Input: path to the Truth.txt file
	Output: pandas dataframe'''
	true_file = ruta
	names_true =['FileName', 'Gender', 'Age Group']
	df_true = pd.read_csv(true_file,sep=',',encoding='utf-8',header=0, names=names_true,index_col=0,engine='python')
	return df_true

def remove_punc(lista, dataframe):
	'''This function removes punctuation marks from the text.
	Input: pandas dataframe column with the text, dataframe.
	Output: dataframe without punctuation marks'''
	for item in lista:
		dataframe[item]=dataframe[item].str.replace('[^\w\s]','')
	return dataframe

if sys.argv[1]=='train':
	if len(sys.argv) != 4:
		print('Error. Usage: python Genre.py train_path test_path output_path')

train_path=sys.argv[1]
test_path=sys.argv[2]
output_path=sys.argv[3]
train_SMS = load_data(train_path)
print('Loading and preparing train data.')
df_true = load_label(train_path+'Truth.csv')
df_train = df_true.copy(deep=True)
for word in ['Gender', 'Age Group']:
	df_train.drop(word, axis=1, inplace=True)
df_train['Texte']=train_SMS
df_train=remove_punc(['Texte'], df_train)
print('Generating probabilities vectors.')
term_set = set()
term_count = {}
for i in range(len(df_train)):
	aux = df_train.iloc[i]['Texte'].split()
	for item in aux:
		if item in term_set:
			pass
		else:
			term_set.add(item)
		if item in term_count:
			term_count[item]+=1
		else:
			term_count[item]=1
for clave in term_count:
	if term_count[clave]<5:
		term_set.remove(clave)
list_term_set = list(term_set)
for item in list_term_set:
	df_train[item]=0
for i in range(len(df_train)):
	aux = df_train.iloc[i]['Texte'].split()
	for item in aux:
		if item in term_set:
			valor = df_train.get_value(df_train.index[i],item)
			valor+=1
			df_train.set_value(df_train.index[i],item,valor)
male_count = {}
total_count = {}
for word in term_set:
	total_count[word]=sum(df_train[word])
for i in range(len(df_train)):
	aux = df_train.iloc[i]
	aux2 = df_true.iloc[i]
	if aux2[0] == 'male':
		for word in term_set:
			if word in male_count:
				male_count[word]= male_count[word] + aux[word]
			else:
				male_count[word]=aux[word]
clase_H = {}
for word in term_set:
	clase_H[word]=male_count[word]/total_count[word]
estadisticos =['MediaH', 'SkewH', 'SDVH']
print('Generating statistical characteristics for train data.')
for i in range(len(df_train)):
	aux = df_train.iloc[i]['Texte'].split()
	LDR_H = [] 
	for word in aux: 
		if word in term_set: 
			LDR_H.append(clase_H[word])
	estadis = {} 
	estadis['MediaH']= np.mean(LDR_H)
	estadis['SkewH']=skew(LDR_H)
	estadis['SDVH']=np.std(LDR_H)
	for item in estadisticos:
		df_train.set_value(df_train.index[i],item,estadis[item])
train_df = df_true.copy(deep=True)
for item in estadisticos:
	train_df[item]=df_train[item]
label_gender=list(train_df['Gender'])
categorias = ['Gender', 'Age Group']
for item in categorias:
	train_df.drop(item, axis=1, inplace=True)
print('Training model.')
#svm=SVC(kernel='linear')
svm=LinearSVC(random_state=0)
svm.fit(train_df, label_gender)
pickle.dump(svm, open(output_path+'gender_model_LinearSVC.sav', 'wb'))

print('Loading test data.')
test_SMS = load_data(test_path)
files_test= [f for f in os.listdir(test_path) if f.endswith('.txt')]
files_test.sort()
df_test=pd.DataFrame(files_test, columns=['Test_Author_Profile_ID']).set_index('Test_Author_Profile_ID')
df_test['Texte']=test_SMS
df_test=remove_punc(['Texte'], df_test)
estadisticos =['MediaH', 'SkewH', 'SDVH']
print('Generating statistical characteristics for test data.')
for i in range(len(df_test)):
	aux = df_test.iloc[i]['Texte'].split()
	LDR_H = []
	for word in aux:
		if word in term_set:
			LDR_H.append(clase_H[word]) 
	estadis = {}
	estadis['MediaH']= np.mean(LDR_H)
	estadis['SkewH']=skew(LDR_H)
	estadis['SDVH']=np.std(LDR_H)
	for item in estadisticos:
		df_test.set_value(df_test.index[i],item,estadis[item])
df_test.drop('Texte', axis=1, inplace=True)
#svm2=pickle.load(open(model_path+'gender_model_LinearSVC.sav', 'rb'))
print('Predicting.')
predicted_gender = svm.predict(df_test)
df_test['Gender']=predicted_gender
for item in estadisticos:
	df_test.drop(item, axis=1, inplace=True)
df_test.to_csv(output_path+'Gender_Predictions.csv')
print('Process completed.')

