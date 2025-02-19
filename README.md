# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load or create sample data
data = {
    'text': ['I love this product', 'This is terrible', 'Amazing experience', 'Disappointing', 'Best ever', 'Worst decision', 'Good quality', 'Bad service'],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)

# Function for text preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing to text data
df['processed_text'] = df['text'].apply(preprocess)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.2, random_state=42)

# Create a pipeline that first vectorizes the text then applies Naive Bayes classifier
model = make_pipeline(
    TfidfVectorizer(),  # Converts text to a matrix of TF-IDF features
    MultinomialNB()     # Naive Bayes classifier
)

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print(classification_report(y_test, y_pred))

# Example usage
new_text = "This is a fantastic product"
preprocessed_text = preprocess(new_text)
predicted_sentiment = model.predict([preprocessed_text])
print(f"Sentiment for '{new_text}': {predicted_sentiment[0]}")
