from logging import debug
from flask import Flask, render_template, request
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import spacy
app = Flask(__name__)

import joblib
model = joblib.load('sentiment_analysis_model.pkl')
vector = joblib.load('vector.pkl')
nlp = spacy.load('en_core_web_sm')

@app.route('/')
def hello():
    return render_template('base.html')

@app.route('/similarity_analysis',methods = ['POST'])
def similarity_analysis():
    ip1 = request.form.get('ip1')
    ip2= request.form.get('ip2')
    
    docs = [ip1,ip2]
    count_vector = CountVectorizer(stop_words='english')
    sparse_matrix = count_vector.fit_transform(docs)

    df = pd.DataFrame(sparse_matrix.toarray(),columns = count_vector.get_feature_names(),index= ['ip1', 'ip2'])
    res = cosine_similarity(df,df)
    res = str(round(res[0][1],4)*100) + "%"
    return render_template('base.html',prediction_text=f'\n the similarity of the entered text is {res}')

if __name__=='__main__':
    app.run(debug=True)