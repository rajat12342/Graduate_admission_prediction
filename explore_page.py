import streamlit as st
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

#This makes sure the data is only loaded once and stored
@st.cache
def load_data():
    df = pandas.read_csv('Admission_Predict.csv', index_col=0)
    return df

df = load_data()

def show_explore_page():
    st.title("Explore the Graduate Admissions dataset")

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
    



