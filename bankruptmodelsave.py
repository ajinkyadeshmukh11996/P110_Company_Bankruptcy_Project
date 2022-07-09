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

#data collection
bankrupt = pd.read_csv('bankruptcy-prevention.csv',sep=';')
bankrupt

#data EDA
bankrupt.columns=bankrupt.columns.str.strip()
label_encoder = preprocessing.LabelEncoder()
bankrupt['class']=label_encoder.fit_transform(bankrupt['class'])

#feature engineering
bankrupt_new= bankrupt[['financial_flexibility','credibility', 'competitiveness', 'class']]
bankrupt_new

#model building
x1= bankrupt_new.iloc[::,0:3:]
y1= bankrupt_new.iloc[::,3:4:]
xtrain,xtest,ytrain,ytest = train_test_split(x1,y1,test_size=0.3,random_state=7)
log_model = LogisticRegression()
log_model.fit(xtrain,ytrain)

#saving the model
filename= 'bankrupt_model.sav'
pickle.dump(log_model,open(filename,'wb'))

#loading the model
load_model = pickle.load(open(filename, 'rb'))
result = load_model.score(xtest, ytest)
print(result)