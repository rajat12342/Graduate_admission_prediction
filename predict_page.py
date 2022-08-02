import streamlit as st
import pandas
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def loadLinear_model():
    with open('linearModel1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data



def loadLogistic_model():
    with open('logisticModel.pkl', 'rb') as file:
        data1 = pickle.load(file)
    return data1






data = loadLinear_model()

data1 = loadLogistic_model()


regression_loaded = data['model']
logisticReg_loaded = data1['model']


def show_predict_page():
    st.title('Graduate school admission prediction')

    st.write("""### Enter information to predict admission chances""")

    tiers = [1,2,3,4,5]

    GreScores = []
    TOEFLscores = []
    SOPscores = [1,1.5,2,2.5,3,3.5,4,4.5,5]
    LORscores = [1,1.5,2,2.5,3,3.5,4,4.5,5]

    for i in range(260,341,1):
        GreScores.append(i)
    for i in range (0,121,1):
        TOEFLscores.append(i)

    research = [0,1]


    GRE = st.selectbox("Select GRE Score: ", GreScores)
    #GRE = st.text_input("Enter GRE SCORE (between 260 and 340): ")
    TOEFL = st.selectbox('Select TOEFL Score: ', TOEFLscores)
    universityRating = st.selectbox("Select undergraduate university tier: 1=very low tier, 5=very high tier", tiers)
    SOP = st.selectbox('Rate your Statement of Purpose: ',SOPscores)
    LOR = st.selectbox('Rate your Letters of Recommendation', LORscores)
    CGPA = st.slider('CGPA (10 point scale): ', 1.0,10.0,0.1)
    Research = st.selectbox('Enter research experience: 0=no research experience, 1=at least some research experience',research)

    

    ok = st.button('Predict admission chance')

    if ok:
        X = np.array([[GRE,TOEFL,universityRating,SOP,LOR,CGPA,Research]])

        
        

        chance = regression_loaded.predict(X)
        chance = chance*100

        binaryOutcome = logisticReg_loaded.predict(X)

        

        if(chance >0 and chance<100):
            #st.subheader(f"The predicted chances of getting into graduate school: {chance[0]:.2f}%")
            st.subheader("The predicted chances of getting into graduate school: {:.2f}%".format(chance[0]))
        else:
            st.subheader('Cannot be ascertained')
        
        if(binaryOutcome ==1 ):
            st.subheader('Outcome predicted: Admitted')
        else:
            st.subheader('Outcome predicted: Rejected')