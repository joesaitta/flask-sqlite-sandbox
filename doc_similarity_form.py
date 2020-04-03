import os, re
import nltk, string
import numpy as np
import sqlite3

from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import gensim

# Load up the pre-trained word embedding model
# pretrained_file = 'GoogleNews-vectors-negative300.bin'
# print('Loading up the W2V model')
# w2v_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_file, binary=True, unicode_errors='ignore')
# print('Model loaded - running the app')

# Establish a connection the the sqlite database
conn = sqlite3.connect('cleaned_text.db')
c = conn.cursor()
# Create the table to hold the data if it doesn't already exist
c.execute('''CREATE TABLE IF NOT EXISTS tblCleanedText
                (id INTEGER PRIMARY KEY, 
                 cleaned_text TEXT)''')
conn.commit()
conn.close()

# Load up the stopwords and corpus for wordnet
try:
    sw = stopwords.words('english')
except:
    nltk.download('stopwords')
    sw = stopwords.words('english')

nltk.download('wordnet')

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

# @app.route('/capstone')
# def capstone():
#     return render_template('doc_sim.html')

@app.route('/_clean_text', methods=['GET', 'POST'])
def clean_text():
    """
    Clean the text series, return the dataframe with lowercase, punctuation and stop words removed
    """
    text = request.form.get('text')
    tokens = clean_and_tokenize(text) 

    # Take the cleaned tokens and write the to the database
    insert_into_db(tokens)

    # Return the json object
    return jsonify(cleaned_doc=tokens)

@app.route('/_doc_sim', methods=['GET', 'POST'])
def doc_sim():
    print('Request form %s' % request.form)
    doc1 = request.form.get('doc1')    
    doc2 = request.form.get('doc2')

    # Clean and tokenize each document
    doc1 = clean_and_tokenize(doc1)
    doc2 = clean_and_tokenize(doc2)

    # Check if either document is empty
    if not doc1 or not doc2:
        return jsonify(doc_similarity='Empty document, cannot compute similarity')

    cos_sim = doc_cos_sim(doc1, doc2)

    print('Document similarity is %s' % cos_sim)

    cos_sim = float(round(cos_sim, 2))
    return jsonify(doc_similarity=cos_sim)

def clean_and_tokenize(text):
    """
    Clean and tokenize a string
    Parameters
    ----------
        text : str
            String to be cleaned and tokenized

    Returns
    -------
        Returns a list of strings from the cleaned document
    """

    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    # Convert everything to lowercase
    text = text.lower()
    
    # Remove punctuation (and bullets)
    regex = re.compile('[%s]' % re.escape(string.punctuation + '’' + u'\uf0b7\u2022,\u2023,\u25E6,\u2043,\u2219'))
    text = regex.sub('', text)
    
    # Tokenize each word in the resume
    tokens = text.split()

    # Remove common stopwords
    tokens = [t for t in tokens if t not in sw ]

    # Get the lemma for each word
    return [lemmatizer.lemmatize(s) for s in tokens]


def insert_into_db(txt, db='cleaned_text.db'):
    """Expecting a list of strings"""
    conn = sqlite3.connect(db)
    c = conn.cursor()

    txt = ' '.join(txt)
    c.execute('''INSERT INTO tblCleanedText (cleaned_text)
                VALUES (?)''', [txt])
    conn.commit()

    # Retrieve the data and print it out
    c.execute('select * From tblCleanedText')
    print(c.fetchall())
    conn.close()


def doc_mean(doc):
    """
    Calculate the centroid of the document vectors
    """
    word_vectors = []
    for word in doc:
        try:
            word_vectors.append(w2v_model.word_vec(word))
        except:
            pass
    word_vectors = np.array(word_vectors)
    return word_vectors.T.mean(axis=1)


def doc_cos_sim(doc1, doc2):
    # Calculate the centroid of each document
    doc1_mean = doc_mean(doc1)
    doc2_mean = doc_mean(doc2)
    
    dot = doc1_mean.dot(doc2_mean)
    doc1_mag = np.linalg.norm(doc1_mean)
    doc2_mag = np.linalg.norm(doc2_mean)
    
    return dot / (doc1_mag * doc2_mag)


if __name__ == '__main__':

    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)