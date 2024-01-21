
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
<p><strong><u>Projenin Hedefi</u></strong></p>
<p style="text-align: justify;">Sahte yorumları tespit etmeye yönelik artan ilgiye rağmen, önceki çalışmalar farklı tüketici deneyimleri gerektiren çeşitli ürünler için sahte yorumları tespit etme kapasitesini araştırmamıştır. Bu sorunların üstesinden gelmek için, en son yapay zeka teknolojilerini kullanarak e-ticaret sitelerindeki sahte yorumları tespit etmek için bir web sitesi önerdik. Sahte yorumları etkili bir şekilde tespit etmek için bir Transformatör (BERT ve Roberta) ve Evrişimsel Sinir Ağlarının (CNN) güçlü yönlerini birleştiren hibrit bir mimari model kullandık.</p>
"""

paragraph2 = """
<p><strong><u>Adanmışlık</u></strong></p>
<p>Rehberim Arş. Gör. Musa DOĞAN'a ve Selçuk Üniversitesi Bilgisayar Mühendisliği Bölümü'nün her bir üyesine teşekkür ederim. Bu çalışmada mükemmellikten daha azını elde etmemde bana yardımcı oldular. Umarım bu site bir bütün olarak toplum için faydalı olur ve tüketicilerin bilinçli kararlar almasına ve çevrimiçi incelemelerin güvenilirliğinin artırılmasına katkıda bulunur. Son olarak, bu web sitesinin kendi özgün ve bağımsız çalışmam olduğunu ve kimsenin telif hakkını ihlal etmediğini veya başka herhangi bir fikri mülkiyet hakkını ihlal etmediğini beyan ederim.</p>
"""

about = """
<p><strong>Adam Mungan&nbsp;</strong>:</p>
<ul>
<li>Bölüm: BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ </li>
<li>Üniversite: SELÇUK ÜNİVERSİTESİ</li>
<li>E-posta: <a href="adammungan@gmail.com" target="_blank" rel="noopener noreferrer">adammungan@gmail.com</a></li>
</ul>
<p><strong>Arş. Gör. Musa DOĞAN ssy</strong> (Danışman):</p>
<ul>
<li>Bölüm: BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ </li>
<li>Üniversite: SELÇUK ÜNİVERSİTESİ</li>
<li>E-posta:&nbsp;<a href="musa.dogan@selcuk.edu.tr" target="_blank" rel="noopener noreferrer">musa.dogan@selcuk.edu.tr</a></li>
</ul>
<p>&nbsp;</p>
"""

p3 = """
<hr/>
<p style="font-family:Calibri (Body); font-size: 14px;"><strong>Adam Mungan</p>
<p>Bilgisayar Mühendisliği</p>
<p>Selçuk Üniversitesi.</p>
<p>📧 adammungan@gmail.com</a></p>
<hr/>

"""


with st.sidebar:
    st.sidebar.image("Asset_2.png" )
    # st.sidebar.image("Logo.png", use_column_width=True )
    st.title(" :blue[Hybrid Spam Checker]")
    st.sidebar.image("Logo.png" )
    
    new_title = '<p style="font-family:Calibri (Body); color:#00B0F0; font-size: 14px;">Designed by</p>'
    st.markdown(new_title, unsafe_allow_html=True)

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
    avg = avg.rename( index={'mean': 'Probability Average'})


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
        vertical_spacing=0.1,column_widths=[0.45, 0.45 , 0.1],
        subplot_titles=("The probabilities of classifiers", "The Probability Average of Classifers", "Average"))
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
            cells_values= [avg[0:1].index, avg[0:1].Spam.values , avg[0:1].NonSpam.values ] , columnwidth = [0.45 , 0.27 , 0.27] 
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
