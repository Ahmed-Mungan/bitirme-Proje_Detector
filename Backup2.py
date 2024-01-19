
# Import Libraries 
#-----------------------------------
# Import Text cleaning function
import streamlit as st
from Cleaning import clean
from tensorflow.keras.preprocessing import sequence
from transformers import BertTokenizer
from transformers import RobertaTokenizer
import pickle
import numpy as np
from keras.models import load_model
from numpy import argmax
from scipy import stats as stt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# -------------------------------

@st.cache (allow_output_mutation = True)
def Import_Models():
    with open('tokenizer.pickle', 'rb') as handle:
        BiLSTM_tokenizer = pickle.load(handle)

    BiLSTM_model = load_model('BiLSTM_CNN.h5')

    Bert_tokenizer = BertTokenizer.from_pretrained('BertTokenizer')
    Bert_model = load_model('BertCnn.h5')

    # Load BERT tokenizer
    Robert_tokenizer = RobertaTokenizer.from_pretrained('RoBertTokenizer')
    RoBert_model = load_model('RoBertCnn.h5')
    return  BiLSTM_tokenizer , BiLSTM_model , Bert_tokenizer , Bert_model , Robert_tokenizer , RoBert_model
# -------------------------------
BiLSTM_tokenizer , BiLSTM_model , Bert_tokenizer , Bert_model , Robert_tokenizer , RoBert_model = Import_Models()

Class = {0:'Non-Spam' , 1 : 'Spam'}
# -------------------------------

def BiLSTM(text):
    sequences = BiLSTM_tokenizer.texts_to_sequences(text)
    sequences = sequence.pad_sequences(sequences, maxlen=50)
    Result = BiLSTM_model.predict(sequences)
    return Result  

def BertCNN (text):
    Bert_Sequences = Bert_tokenizer(text, padding= 'max_length' , truncation=True, max_length=80)
    Result = Bert_model.predict(Bert_Sequences['input_ids'])
    return Result  

def RoBertCNN (text):
    Robert_Sequences = Robert_tokenizer(text, padding= 'max_length' , truncation=True, max_length=80)
    Result = RoBert_model.predict(Robert_Sequences['input_ids'])
    return Result

def calssification (text):
    # Classification using classifier
    BiLSTM_Result = BiLSTM(text)
    Bert_Result = BertCNN (text)
    RoBertCNN_Result = RoBertCNN(text)

    # Calculate the result
    BiLSTM_output=np.argmax(BiLSTM_Result,axis=1)
    Bert_output=np.argmax(Bert_Result,axis=1)
    RoBertCNN_output=np.argmax(RoBertCNN_Result,axis=1)
    Result = Class[stt.mode ([BiLSTM_output ,Bert_output ,RoBertCNN_output  ])[0][0]]


    # Creat the dataframe
    BiLSTM_H = round (BiLSTM_Result[0][0]*100 ,1) 
    Bert_H = round (Bert_Result[0][0]*100,1) 
    Roberta_H = round(RoBertCNN_Result[0][0]*100,1) 
          
    BiLSTM_S = round (BiLSTM_Result[0][1]*100,1) 
    Bert_S = round (Bert_Result[0][1]*100,1)
    Roberta_S = round (RoBertCNN_Result[0][1]*100,1)

    if Result == "Spam" :
        Spam_values = [Bert_S,  BiLSTM_S , Roberta_S]
        Spam_values = sorted (Spam_values)
        Non_Spam_values = [round ( abs (100-i) ,1)  for i in Spam_values]

    else:
        Non_Spam_values = [Bert_H   , BiLSTM_H , Roberta_H ]
        Non_Spam_values =  sorted (Non_Spam_values)
        Spam_values =     [round (abs (100-i),1)  for i in Non_Spam_values]
        

    ix = ['CNN' , 'Hybrid RoBerta-CNN' ,'Hybrid Bert-CNN']
    columns = {'NonSpam':Non_Spam_values , 'Spam':Spam_values}

    df = pd.DataFrame(columns  , index=ix )

    # df['NonSpam'] = df['NonSpam'].apply( lambda x : str(x) + '%')
    # df['Spam'] = df['Spam'].apply( lambda x : str(x) + '%')
    df.index.name = 'Classifier'

    Spam_avg = { "NonSpam":[ sum (Non_Spam_values) / 3] , "Spam": [ sum(Spam_values) / 3] }
    avg = pd.DataFrame.from_dict(Spam_avg).describe().iloc[1:2]

    # avg['Non_Spam'] = avg['Non_Spam'].apply( lambda x : str(x) + '%')
    # avg['Spam'] = avg['Spam'].apply( lambda x : str(x) + '%')


    return (Result , df , avg)
    #print (df)

#---------------------------------------------------------------------

paragraph = """
<p><strong><u>The Goal of Designing This Website</u></strong></p>
<p style="text-align: justify;">Despite the growing interest in detecting false reviews, prior studies have not explored the capacity to detect fake reviews for diverse products, which require distinct consumer experiences. To overcome these problems, we proposed a website to detect fake reviews on e-commerce sites using the latest artificial intelligence technologies. We have employed a hybrid architecture model that combines the strengths of a Transformer (BERT and Roberta) and Convolutional Neural Networks (CNN) to effectively detect fake reviews.</p>
"""

paragraph2 = """
<p><strong><u>Dedication</u></strong></p>
<p>A very special thanks to my guide Prof. Dr. Hiren Joshi, and each member of the Department of Computer Science, Gujarat University. They helped me achieve nothing less than excellence in this work. I hope that this site will be useful to society as a whole and contribute to helping consumers make informed decisions and improving the credibility of online reviews. In the end, I declare that this website is my own original and independent work and does not infringe upon anyone&rsquo;s copyright or violate any other intellectual property rights.</p>
"""

about = """
<p><strong>Maysara Mazin Alsaad&nbsp;</strong>(PhD Candidate):</p>
<ul>
<li>Department: Computer Science</li>
<li>University: Gujarat University, Ahmedabad, India.</li>
<li>Email: <a href="mailto:maysara@gujaratuniversity.ac.in" target="_blank" rel="noopener noreferrer">maysara@gujaratuniversity.ac.in</a></li>
</ul>
<p><strong>Prof. Dr. Hiren Joshi</strong> (Guide):</p>
<ul>
<li>Department: Computer Science</li>
<li>University: Gujarat University, Ahmedabad, India.</li>
<li>Email:&nbsp;<a href="mailto:hdjoshi@gujaratuniversity.ac.in" target="_blank" rel="noopener noreferrer">hdjoshi@gujaratuniversity.ac.in</a></li>
</ul>
<p>&nbsp;</p>
"""

p3 = """
<hr/>
<p style="font-family:Calibri (Body); font-size: 14px;"><strong>Maysara Mazin Alsaad </strong>(PhD Candidate)</p>
<p>Department of Computer Science</p>
<p>Gujarat University, Ahmedabad, India.</p>
<p>📧 maysara@gujaratuniversity.ac.in</a></p>
<p>📞 +974 66457667</p>
<hr/>

"""


with st.sidebar:
    st.sidebar.image("Asset_2.png" )
    # st.sidebar.image("Logo.png", use_column_width=True )
    st.title(" :blue[Hybrid Spam Checker]")
    st.sidebar.image("Logo.png" )


    # st.caption("Maysara Mazin Alsaad (PhD Candidate)")
    # st.write(paragraph)
    st.write(about, unsafe_allow_html=True)
    st.write(paragraph, unsafe_allow_html=True)
    st.write(paragraph2, unsafe_allow_html=True)
    st.write(p3, unsafe_allow_html=True)

prompt = st.chat_input("Say something")
if prompt:

    new_title = '<p style="font-family:Calibri (Body); color:#F8931F; font-size: 18px;">Input: </p>'
    st.markdown(new_title, unsafe_allow_html=True )
    st.write(prompt)


    CleanedText = clean (prompt)
    new_title = '<p style="font-family:Calibri (Body); color:#F8931F; font-size: 18px;">Cleaned Text: </p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(CleanedText)

    new_title = '<p style="font-family:Calibri (Body); color:#F8931F; font-size: 18px;">Result: </p>'
    st.markdown(new_title, unsafe_allow_html=True)



    CL , df, avg2 = calssification([CleanedText])
    avg = df.describe().loc[['mean' , 'min' , 'max']]

    df['NonSpam'] = df['NonSpam'].apply( lambda x : str(round (x,1)) + '%')
    df['Spam'] = df['Spam'].apply( lambda x : str(round (x,1)) + '%')
    avg['NonSpam'] = avg['NonSpam'].apply( lambda x : str(round (x,1)) + '%')
    avg['Spam'] = avg['Spam'].apply( lambda x : str(round (x ,1)) + '%')


    if CL == "Spam":
        st.error('Spam', icon="⛔")  
    else:
        st.success('Non-Spam', icon="✅")

    new_title = '<p style="font-family:Calibri (Body); color:#F8931F; font-size: 18px;">Probabilities & Statistics : </p>'
    st.markdown(new_title, unsafe_allow_html=True)

    colors = ["#16C60C", "#F03A17"]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "table"}, {"type": "table"} , {"type": "bar"}]], 
        vertical_spacing=0.1,column_widths=[0.6, 0.3 , 0.1],
        subplot_titles=("The probabilities of classifiers", "Statistics", "Average"))
    ##########################################
    fig.add_trace(
        go.Table(
            header_values= ["Classifier" , "Sapm" , "Non-Spam"],
            cells_values= [df.index ,df.Spam.values , df.NonSpam.values ] , columnwidth = [0.55 , 0.2 , 0.25]
        ),  row=1, col=1)
    ##########################################
    fig.add_trace(
        go.Table(
            header_values= ["Statistic" , "Sapm" , "Non-Spam"],
            cells_values= [avg.index, avg.Spam.values , avg.NonSpam.values ] , columnwidth = [0.33 , 0.33 , 0.33] 
        ),  row=1, col=2)

    ##########################################

    for r, c in zip(['NonSpam' , 'Spam'], colors):
        fig.add_trace(
            go.Bar(x= avg[0:1][r].index , y= avg[r].values , name= r, marker_color=c ,) 
            ,row=1, col=3
        )

    fig.update_layout(width = 900 ,
        height=320,
        showlegend=False,
        barmode="stack") # title_text="Bitcoin mining stats for 180 days" ,
    # fig.show()

    st.plotly_chart(fig, theme=None)