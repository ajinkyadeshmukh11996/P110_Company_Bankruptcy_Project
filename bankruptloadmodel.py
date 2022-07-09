import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import ppscore as pps
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import pickle
from statistics import mode
import streamlit as st

st.title('Bankrupt Model')
#data collection
bankrupt = pd.read_csv('bankruptcy-prevention.csv',sep=';')
bankrupt

#data EDA
bankrupt.columns=bankrupt.columns.str.strip()
label_encoder = preprocessing.LabelEncoder()
bankrupt['class']=label_encoder.fit_transform(bankrupt['class'])

#feature engineering
bankrupt_new= bankrupt[['financial_flexibility','credibility', 'competitiveness', 'class']]

st.subheader('Input paramters')
st.write(bankrupt.iloc[::,2:5])
x1= bankrupt_new.iloc[::,0:3:]
y1= bankrupt_new.iloc[::,3:4:]

#loading the model
filename= 'bankrupt_model.sav'
load_model = pickle.load(open(filename, 'rb'))
result = load_model.score(x1, y1)
predictions =load_model.predict(x1)
print(predictions)
st.subheader('Predictions')
st.write(predictions)
predi_probability = load_model.predict_proba(x1)
print(predi_probability)
st.subheader('Prediction Probability')
st.write(predi_probability)

bankrupt_output = pd.concat([bankrupt,pd.DataFrame(predi_probability)],axis=1)
bankrupt_output.to_excel('bankrupt_output.xlsx')