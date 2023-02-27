import pandas as pd
import string
import nltk
import joblib
import numpy as np
import lightgbm

from sklearn.ensemble import BaggingClassifier
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

w2v_synopsis = joblib.load('./ml_model/w2v_model_synopsis.pkl')
w2v_title = joblib.load('./ml_model/w2v_model_title.pkl')
w2v_studio = joblib.load('./ml_model/w2v_model_studio.pkl')
w2v_producer = joblib.load('./ml_model/w2v_model_producer.pkl')
clf = joblib.load('./ml_model/bagging_clf_final.pkl')

gender_list = ['Horror', 'Historical', 'Adventure', 'Sci-Fi', 'Shounen', 'Seinen', 'Super Power', 'Kids', 'Samurai', 'Mystery', 'Police', 'Yuri', 'Romance', 'Space', 'Game', 'Shoujo', 'Martial Arts', 'Shounen Ai', 'Josei', 'Military', 'Psychological', 'Thriller', 'Parody', 'Music', 'Mecha', 'Yaoi', 'Supernatural', 'Demons', 'Sports', 'Action', 'Shoujo Ai', 'Dementia', 'Harem', 'Hentai', 'Cars', 'Slice of Life', 'Drama', 'Fantasy', 'Magic', 'Ecchi', 'Comedy']

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
  stop_words = set(stopwords.words('english')) 
  return ' '.join([word for word in text.split() if word not in stop_words])

def stem_text(text):
  stemmer = SnowballStemmer('english')
  return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_text(text):
  lemmatizer = WordNetLemmatizer()
  return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Convert each synopsis into a numeric vector using the trained Word2Vec model
def vectorize_synopsis(synopsis):
    tokens = synopsis.split()
    vectors = [w2v_synopsis.wv.get_vector(token) for token in tokens if token in w2v_synopsis.wv.key_to_index]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

def vectorize_title(title):
    tokens = title.split()
    vectors = [w2v_title.wv.get_vector(token) for token in tokens if token in w2v_title.wv.key_to_index]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

def vectorize_producer(producer):
    tokens = producer.split()
    vectors = [w2v_producer.wv.get_vector(token) for token in tokens if token in w2v_producer.wv.key_to_index]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

def vectorize_studio(studio):
    tokens = studio.split()
    vectors = [w2v_studio.wv.get_vector(token) for token in tokens if token in w2v_studio.wv.key_to_index]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

def add_genre_columns(df):
    for genre in gender_list:
        df[genre] = 0
        
    #df['Gender'] = df['Gender'].apply(eval)
    present_gender = set(g for genres in df['Gender'] for g in genres)

    for genre in present_gender:
        df[genre] = df['Gender'].apply(lambda x: 1 if genre in x else 0)
        
    return df

def process_input(df):
    type_mapping = {'TV': 0, 'Movie': 1, 'OVA': 2, 'Special': 3, 'ONA': 4, 'Music': 5}
    source_mapping = {'Original': 0, 'Manga': 1, 'Light novel': 2, 'Game': 3, 'Visual novel': 4, '4-koma manga': 5, 'Novel': 6, 'Unknown': 7, 'Other': 8, 'Picture book': 9, 'Web manga': 10, 'Music': 11, 'Book': 12, 'Card game': 13, 'Radio': 14, 'Digital manga': 15}
    df['Type'] = df['Type'].replace(type_mapping)
    df['Source'] = df['Source'].replace(source_mapping)
    
    df = add_genre_columns(df)
    
    df['preprocessed_synopsis'] = df['Synopsis'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['preprocessed_synopsis'] = df['preprocessed_synopsis'].apply(remove_punctuation)
    df['preprocessed_synopsis'] = df['preprocessed_synopsis'].apply(remove_stopwords)
    df['preprocessed_synopsis'] = df['preprocessed_synopsis'].apply(stem_text)
    df['preprocessed_synopsis'] = df['preprocessed_synopsis'].apply(lemmatize_text)
    
    df['preprocessed_Title'] = df['Title'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['preprocessed_Title'] = df['preprocessed_Title'].apply(remove_punctuation)
    df['preprocessed_Title'] = df['preprocessed_Title'].apply(remove_stopwords)
    df['preprocessed_Title'] = df['preprocessed_Title'].apply(stem_text)
    df['preprocessed_Title'] = df['preprocessed_Title'].apply(lemmatize_text)
    
    df['preprocessed_Producer'] = df['Producer'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['preprocessed_Producer'] = df['preprocessed_Producer'].apply(remove_punctuation)
    df['preprocessed_Producer'] = df['preprocessed_Producer'].apply(remove_stopwords)
    df['preprocessed_Producer'] = df['preprocessed_Producer'].apply(stem_text)
    df['preprocessed_Producer'] = df['preprocessed_Producer'].apply(lemmatize_text)
    
    df['preprocessed_Studio'] = df['Studio'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['preprocessed_Studio'] = df['preprocessed_Studio'].apply(remove_punctuation)
    df['preprocessed_Studio'] = df['preprocessed_Studio'].apply(remove_stopwords)
    df['preprocessed_Studio'] = df['preprocessed_Studio'].apply(stem_text)
    df['preprocessed_Studio'] = df['preprocessed_Studio'].apply(lemmatize_text)
    
    df['synopsis_vectors'] = df['preprocessed_synopsis'].apply(vectorize_synopsis)
    df['title_vectors'] = df['preprocessed_Title'].apply(vectorize_title)
    df['producer_vectors'] = df['preprocessed_Producer'].apply(vectorize_producer)
    df['studio_vectors'] = df['preprocessed_Studio'].apply(vectorize_studio)
    
    df = df.drop(['Synopsis', 'preprocessed_synopsis', 'Title', 'preprocessed_Title', 'Producer', 'preprocessed_Producer', 'Studio', 'preprocessed_Studio'], axis=1)
    
    df['synopsis_vectors'] = df['synopsis_vectors'].apply(lambda x: np.array(x))
    df['title_vectors'] = df['title_vectors'].apply(lambda x: np.array(x))
    df['producer_vectors'] = df['producer_vectors'].apply(lambda x: np.array(x))
    df['studio_vectors'] = df['studio_vectors'].apply(lambda x: np.array(x))
            
    X = np.concatenate([df['synopsis_vectors'].values.tolist(),
                    df['title_vectors'].values.tolist(), 
                    df['Type'].values.reshape(-1,1),
                    df['producer_vectors'].values.tolist(), 
                    df['studio_vectors'].values.tolist(), 
                    df['Source'].values.reshape(-1,1),
                    df['Horror'].values.reshape(-1,1),
                    df['Historical'].values.reshape(-1,1),
                    df['Adventure'].values.reshape(-1,1),
                    df['Sci-Fi'].values.reshape(-1,1),
                    df['Shounen'].values.reshape(-1,1),
                    df['Seinen'].values.reshape(-1,1),
                    df['Super Power'].values.reshape(-1,1), 
                    df['Kids'].values.reshape(-1,1),
                    df['Samurai'].values.reshape(-1,1),
                    df['Mystery'].values.reshape(-1,1),
                    df['Police'].values.reshape(-1,1),  
                    df['Yuri'].values.reshape(-1,1),
                    df['Romance'].values.reshape(-1,1),
                    df['Space'].values.reshape(-1,1),
                    df['Game'].values.reshape(-1,1),
                    df['Shoujo'].values.reshape(-1,1),
                    df['Martial Arts'].values.reshape(-1,1),
                    df['Shounen Ai'].values.reshape(-1,1),
                    df['Josei'].values.reshape(-1,1),
                    df['Military'].values.reshape(-1,1),
                    df['Psychological'].values.reshape(-1,1),
                    df['Thriller'].values.reshape(-1,1),
                    df['Parody'].values.reshape(-1,1),
                    df['Music'].values.reshape(-1,1), 
                    df['Mecha'].values.reshape(-1,1),
                    df['Yaoi'].values.reshape(-1,1),
                    df['Supernatural'].values.reshape(-1,1),
                    df['Demons'].values.reshape(-1,1),
                    df['Sports'].values.reshape(-1,1), 
                    df['Action'].values.reshape(-1,1),
                    df['Shoujo Ai'].values.reshape(-1,1),
                    df['Dementia'].values.reshape(-1,1),
                    df['Harem'].values.reshape(-1,1),
                    df['Hentai'].values.reshape(-1,1),
                    df['Cars'].values.reshape(-1,1),
                    df['Slice of Life'].values.reshape(-1,1),
                    df['Drama'].values.reshape(-1,1),
                    df['Fantasy'].values.reshape(-1,1),
                    df['Magic'].values.reshape(-1,1),
                    df['Ecchi'].values.reshape(-1,1),               
                    df['Comedy'].values.reshape(-1,1),],
                    axis=1)
    
    return int(clf.predict(X))




