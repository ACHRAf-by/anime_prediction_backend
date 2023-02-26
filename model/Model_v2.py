import nltk
import gensim
import string
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer

pd.options.mode.chained_assignment = None

anime_df = pd.read_csv('Anime_data.csv')

anime_df.shape

anime_df.head()

anime_df.dtypes

anime_df.columns

anime_df.isna().sum()

# Handling missing values
anime_df = anime_df[(anime_df['Rating'].notnull()) & (anime_df['Source'].notnull()) & (anime_df['Aired'].notnull()) & (anime_df['Synopsis'].notnull())]

anime_df.loc[anime_df['Producer'].isna(), 'Producer'] = 'unknown'
anime_df.loc[anime_df['Studio'].isna(), 'Studio'] = 'unknown'

# Converting categorical variables into numerical values
anime_df['Type'] = pd.factorize(anime_df['Type'])[0]
anime_df['Source'] = pd.factorize(anime_df['Source'])[0]
anime_df['Producer'] = pd.factorize(anime_df['Producer'])[0]
anime_df['Studio'] = pd.factorize(anime_df['Studio'])[0]
anime_df['Title'] = pd.factorize(anime_df['Title'])[0]

# Dropping irrelevant columns
anime_df = anime_df.drop(['Anime_id', 'Aired', 'Link', 'Episodes'], axis=1)

# Replace missing values in the Genre column with an empty list
anime_df['Genre'].fillna(value='[]', inplace=True)

# Converting the genre column to a list of genres
anime_df['Genre'] = anime_df['Genre'].apply(eval)

# Performing one-hot encoding on the genre column
genres = set(g for genres in anime_df['Genre'] for g in genres)
for genre in genres:
    anime_df[genre] = anime_df['Genre'].apply(lambda x: 1 if genre in x else 0)

# Dropping the original genre column
anime_df.drop('Genre', axis=1, inplace=True)

anime_df.head()

# Define the pre-processing functions
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

# Preprocess the Synopsis of the anime
anime_df['preprocessed_synopsis'] = anime_df['Synopsis'].apply(lambda x: x.lower() if isinstance(x, str) else x)
anime_df['preprocessed_synopsis'] = anime_df['preprocessed_synopsis'].apply(remove_punctuation)
anime_df['preprocessed_synopsis'] = anime_df['preprocessed_synopsis'].apply(remove_stopwords)
anime_df['preprocessed_synopsis'] = anime_df['preprocessed_synopsis'].apply(stem_text)
anime_df['preprocessed_synopsis'] = anime_df['preprocessed_synopsis'].apply(lemmatize_text)

# Convert the preprocessed synopsis into a list of tokenized sentences
sentences = [synopsis.split() for synopsis in anime_df['preprocessed_synopsis']]

# Train a Word2Vec model on the tokenized sentences
w2v_model = gensim.models.Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

# Convert each synopsis into a numeric vector using the trained Word2Vec model
def vectorize_synopsis(synopsis):
    tokens = synopsis.split()
    vectors = [w2v_model.wv.get_vector(token) for token in tokens if token in w2v_model.wv.key_to_index]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

anime_df['synopsis_vectors'] = anime_df['preprocessed_synopsis'].apply(vectorize_synopsis)


anime_df.head()

anime_df['Type'] = pd.factorize(anime_df['Type'])[0]
anime_df['Source'] = pd.factorize(anime_df['Source'])[0]
anime_df['Producer'] = pd.factorize(anime_df['Producer'])[0]
anime_df['Studio'] = pd.factorize(anime_df['Studio'])[0]
anime_df['Title'] = pd.factorize(anime_df['Title'])[0]

# Dropping irrelevant columns
anime_df = anime_df.drop(['Synopsis', 'preprocessed_synopsis'], axis=1)

anime_df['synopsis_vectors'] = anime_df['synopsis_vectors'].apply(lambda x: np.array(x))

# Concatenate vectors with the other columns 
X = np.concatenate([anime_df['synopsis_vectors'].values.tolist(),
                    anime_df['Title'].values.reshape(-1,1), 
                    anime_df['Type'].values.reshape(-1,1),
                    anime_df['Producer'].values.reshape(-1,1), 
                    anime_df['Studio'].values.reshape(-1,1), 
                    anime_df['ScoredBy'].values.reshape(-1,1), 
                    anime_df['Popularity'].values.reshape(-1,1), 
                    anime_df['Members'].values.reshape(-1,1),
                    anime_df['Source'].values.reshape(-1,1),
                    anime_df['Horror'].values.reshape(-1,1),
                    anime_df['Historical'].values.reshape(-1,1),
                    anime_df['Adventure'].values.reshape(-1,1),
                    anime_df['Sci-Fi'].values.reshape(-1,1),
                    anime_df['Shounen'].values.reshape(-1,1),
                    anime_df['Seinen'].values.reshape(-1,1),
                    anime_df['Super Power'].values.reshape(-1,1), 
                    anime_df['Kids'].values.reshape(-1,1),
                    anime_df['Samurai'].values.reshape(-1,1),
                    anime_df['Mystery'].values.reshape(-1,1),
                    anime_df['Police'].values.reshape(-1,1),  
                    anime_df['Yuri'].values.reshape(-1,1),
                    anime_df['Romance'].values.reshape(-1,1),
                    anime_df['Space'].values.reshape(-1,1),
                    anime_df['Game'].values.reshape(-1,1),
                    anime_df['Shoujo'].values.reshape(-1,1),
                    anime_df['Martial Arts'].values.reshape(-1,1),
                    anime_df['Shounen Ai'].values.reshape(-1,1),
                    anime_df['Josei'].values.reshape(-1,1),
                    anime_df['Military'].values.reshape(-1,1),
                    anime_df['Psychological'].values.reshape(-1,1),
                    anime_df['Thriller'].values.reshape(-1,1),
                    anime_df['Parody'].values.reshape(-1,1),
                    anime_df['Music'].values.reshape(-1,1), 
                    anime_df['Mecha'].values.reshape(-1,1),
                    anime_df['Yaoi'].values.reshape(-1,1),
                    anime_df['Supernatural'].values.reshape(-1,1),
                    anime_df['Demons'].values.reshape(-1,1),
                    anime_df['Sports'].values.reshape(-1,1), 
                    anime_df['Action'].values.reshape(-1,1),
                    anime_df['Shoujo Ai'].values.reshape(-1,1),
                    anime_df['Dementia'].values.reshape(-1,1),
                    anime_df['Harem'].values.reshape(-1,1),
                    anime_df['Hentai'].values.reshape(-1,1),
                    anime_df['Cars'].values.reshape(-1,1),
                    anime_df['Slice of Life'].values.reshape(-1,1),
                    anime_df['Drama'].values.reshape(-1,1),
                    anime_df['Fantasy'].values.reshape(-1,1),
                    anime_df['Magic'].values.reshape(-1,1),
                    anime_df['Ecchi'].values.reshape(-1,1),               
                    anime_df['Comedy'].values.reshape(-1,1),],
                    axis=1)

# Splitting the dataset into features and target variable
#  = anime_df.drop(['Rating'], axis=1)
# y = X['Rating']

y = anime_df['Rating'].values

from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the selected model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predicting the ratings on the test set
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

# Evaluating the model's performance using r2_score
r2 = r2_score(y_test, y_pred)
print("R2 score: ", r2)

# Evaluating the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

# Evaluating the model's performance using root_mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)

y_pred_df = pd.DataFrame(y_pred, columns=['predicted_rating'])
print(y_pred_df.head())

