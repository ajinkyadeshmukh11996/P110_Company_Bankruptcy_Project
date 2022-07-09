#Dumping the model
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import ppscore as pps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
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
import streamlit as st

st.title('P110 model')
#Doing the EDA part
recom = pd.read_excel('recommendations_trial.xlsx')
x= recom

st.subheader('Input parameters')
st.write(recom)
#loading the model
filename= 'bankrupt_model.sav'
load_model = pickle.load(open(filename, 'rb'))
predictions =load_model.predict(x)
print(predictions)
st.subheader('predictions')
st.write(predictions)
predi_probability =load_model.predict_proba(x)
st.subheader('Prediction probability')
st.write(predi_probability)
output2=pd.concat([recom,pd.DataFrame(predictions)],axis=1)
output2.to_excel('recom1.xlsx')