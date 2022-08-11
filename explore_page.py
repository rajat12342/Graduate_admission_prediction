import streamlit as st
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns
from predict_page import loadLinear_model
from predict_page import loadLogistic_model

#This makes sure the data is only loaded once and stored
@st.cache
def load_data():
    df = pandas.read_csv('Admission_Predict.csv', index_col=0)
    return df

df = load_data()

linearModel = loadLinear_model()
logisticModel = loadLogistic_model()


def show_explore_page():
    st.title("Explore the UCLA Graduate Admissions dataset")

    fig = plt.figure()

    importance = linearModel['model'].coef_
    Scores = []
    List = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    for i,v in enumerate(importance):
        Scores.append(v)
    
    ScoresS = pandas.Series(Scores, name='Importance')
    ListS = pandas.Series(List, name='Features')
    Features_df_lin = pandas.concat([ScoresS, ListS], axis=1)

    plt.barh([1,2,3,4,5,6,7], Features_df_lin['Importance'])
    plt.yticks([1,2,3,4,5,6,7], Features_df_lin['Features'])
    plt.title('Linear Regression Feature Coefficients')
    plt.xlabel('Coefficients')
    st.pyplot(fig)

    fig = plt.figure()

    importance_log = logisticModel['model'].coef_[0]

    Scores_log = []

    for i,v in enumerate(importance_log):
        Scores_log.append(v)
    
    Scores_logS = pandas.Series(Scores_log, name='Importance')

    Features_df_log = pandas.concat([Scores_logS, ListS], axis=1)

    plt.barh([1,2,3,4,5,6,7], Features_df_log['Importance'])
    plt.yticks([1,2,3,4,5,6,7], Features_df_log['Features'])
    plt.xlabel('Coefficients')

    plt.title('Logistic Regression Feature Coefficients')    
    st.pyplot(fig)

    fig = plt.figure(figsize=(7,7))

    plot = sns.histplot(x=df['GRE Score'], kde = True)
    plt.title("GRE Score distribution")
    plt.ylabel('Number of applicants')
    #Create the figure above and get the current figure through gcf
    st.pyplot(fig)


    fig = plt.figure(figsize=(7,7))
    plot = sns.histplot(x=df['CGPA'], kde = True)
    plt.ylabel('Number of applicants')
    plt.title("CGPA distribution")
    st.pyplot(fig)

    fig = plt.figure(figsize=(7,7))
    plot = sns.histplot(x=df['TOEFL Score'], kde= True)
    plt.ylabel('Number of applicants')
    plt.title('TOEFL scores distribution')
    st.pyplot(fig)

    fig = plt.figure(figsize=(7,7))
    plot = sns.histplot(x=df['SOP'])
    plt.ylabel('Number of applicants')
    plt.title('Statement of purpose ratings distribution')
    st.pyplot(fig)

    fig = plt.figure(figsize=(7,7))
    plot = sns.histplot(x=df['LOR '])
    plt.ylabel('Number of applicants')
    plt.title('Letters of recommendation ratings distribution')
    st.pyplot(fig)


    #Look at various counts of university ratings and research experience
    fig = plt.figure(figsize=(7,7))
    plot = sns.countplot(x=df['University Rating'])
    plt.title('University ratings count')
    plt.ylabel('Number of applicants')
    st.pyplot(fig)

    fig = plt.figure(figsize=(7,7))
    #fig, ax = plt.subplots(figsize=(7,7))
    plot = sns.countplot(x=df['Research'])
    plt.ylabel('Number of applicants')

    plot.set_xticklabels(['No research experience', 'Has research experience'])
    plt.title('Research experience count')
    st.pyplot(fig)


    fig = plt.figure(figsize=(7,7))
    plot = sns.regplot(x='GRE Score', y='TOEFL Score', data = df)
    plt.title('GRE Scores vs TOEFL Scores')
    st.pyplot(plt.gcf())

    fig = plt.figure(figsize=(7,7))
    plot = sns.lmplot(x='GRE Score', y='CGPA', data=df, hue='Research')
    plt.title('GRE Scores vs CGPA seperated by research experience')
    st.pyplot(plt.gcf())

    fig = plt.figure(figsize=(7,7))
    plot = sns.lmplot(x='GRE Score', y='CGPA', data = df, hue='LOR ')
    plt.title('GRE scores vs CGPA seperated by Letters of recommendation ratings')
    st.pyplot(plt.gcf())

    fig = plt.figure(figsize=(7,7))
    plot = sns.lmplot(x='GRE Score', y='CGPA', data=df, hue='SOP')
    plt.title('GRE Scores vs CGPA seperated by statement of purpose ratings')
    st.pyplot(plt.gcf())
    
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    corr = df.corr()
    fig = plt.figure(figsize=(6,6))
    plt.title('Correlation matrix')
    plot = sns.heatmap(corr, annot=True, cmap = colormap, fmt='.2f', linewidths=0.5)
    st.pyplot(plt.gcf())


