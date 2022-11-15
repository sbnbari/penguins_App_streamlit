import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, recall_score, precision_score

#make containers
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Penguins Ki App by Sabeen')
    st.text('In this project we will work on penguins data')
    
    #import dataset
    df = sns.load_dataset('penguins')
    
    df.dropna(subset = ['body_mass_g', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'], inplace = True)
    
    #dropping island categorical value
    df.drop("island", axis=1, inplace = True)
    
    #changed sex in 1 and 0
    df['sex']= df['sex'].replace('Male',1)
    df['sex']= df['sex'].replace('Female',0)
    
    #changing values categorical to numerical
    df['species']= df['species'].replace('Adelie',1)
    df['species']= df['species'].replace('Chinstrap',2)
    df['species']= df['species'].replace('Gentoo',3)
    
    st.write(df.head(10))
    
    #plotting bar charts
    st.subheader('Penguins Gender Counts')
    st.bar_chart(df['sex'].value_counts())
    st.subheader('Penguins Species')
    st.bar_chart(df['species'].value_counts())
    st.subheader('Penguins Body Mass in grams')
    st.bar_chart(df['body_mass_g'].sample(10))
    #st.subheader('Penguins Island')
    #st.bar_chart(df['island'].value_counts())
    
    

with datasets:
    st.header('Analysis of Data')
    st.text('Working on penguins dataset')

with features:
    st.header('These are our app features:')
    st.text('We are going to add lots of features')
    st.markdown('1. **Feature 1:** Random Forest Regressor')
    st.markdown('2. **Feature 2:** Turned categorical dat in numeric')
    st.markdown('3. **Feature 3:** Mean Squared Error')

with model_training:
    st.header('penguins ka kiya bana? Model Training')
    st.text('we can make changes in our parameters') 
    #making columns
    input, display = st.columns(2)
    
    #pehle column mei selection points ho
    max_depth = input.slider('how many Penguins?', min_value=10, max_value=100, step=5 )

#for random forest we need n estimators
n_estimators = input.selectbox('how many trees should be there in RF?', options=[50,100,200,300,'No limit'])    


#adding list of features
input.write(df.columns)

#input features from user

input_features = input.text_input('Which features we should use?')


#machine learning model          

model = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)


#define X and y

X = df[[input_features]]
y = df[['body_mass_g']]

#fit our model
model.fit(X,y)
pred = model.predict(y)

#display metrices

display.subheader('Mean absolute error of model is:')
display.write(mean_absolute_error(y,pred))

display.subheader('Mean squared error of model is:')
display.write(mean_squared_error(y,pred))

display.subheader('R squared score of model is:')
display.write(r2_score(y,pred))
