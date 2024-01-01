import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import  
import pickle

df=pd.read_csv("sensitive_dataset.csv")

df["Contact"]=df["Contact"].replace('+91',"")

x = df.iloc[:,100]
print(x)



#x_train,x_test,y_train,y_test=train_test_split()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

