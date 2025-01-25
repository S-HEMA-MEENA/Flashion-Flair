import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class DataPreparation:
    def __init__(self, df, feature_column, label_column, test_size=0.2, random_state=42):
        self.df = df
        self.feature_column = feature_column
        self.label_column = label_column
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        X = self.df[self.feature_column]
        y = self.df[self.label_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
    

class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean(self, doc):
        doc = doc.replace("READ MORE", " ")
        doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
        doc = doc.lower()
        tokens = nltk.word_tokenize(doc)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in self.stop_words]
        return " ".join(filtered_tokens)

    def preprocess_dataset(self, documents):
        return [self.clean(doc) for doc in documents]

class ModelTrainer:
    def __init__(self, classifier=None):
        self.classifier = classifier if classifier else RandomForestClassifier()
    
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X):
        return self.classifier.predict(X)
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        return y_train_pred, y_test_pred

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(how='all', axis=1)
    return df

def generate_count_plot(df):
    plt.figure(figsize=(10, 6))
    count_plot = sns.countplot(x="Predicted_Label", data=df)
    plt.xlabel('Predicted Label')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Labels')
    return count_plot.figure

def generate_wordclouds(df):
    positive_reviews = df[df['Predicted_Label'] == 1]
    positive_text = ' '.join(review for review in positive_reviews['review'])
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

    negative_reviews = df[df['Predicted_Label'] == 0]
    negative_text = ' '.join(review for review in negative_reviews['review'])
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

    return positive_wordcloud, negative_wordcloud

def generate_classification_reports(y_train, y_train_pred, y_test, y_test_pred):
    train_report = classification_report(y_train, y_train_pred)
    test_report = classification_report(y_test, y_test_pred)
    return train_report, test_report
