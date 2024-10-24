# Install required packages
!pip install nltk scikit-learn pandas numpy matplotlib seaborn

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DocumentComparator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_keyword_frequencies(self, text, top_n=10):
        """Get frequency distribution of keywords"""
        tokens = word_tokenize(self.preprocess_text(text))
        freq_dist = nltk.FreqDist(tokens)
        return dict(freq_dist.most_common(top_n))
    
    def calculate_tfidf_similarity(self, docs):
        """Calculate TF-IDF based similarity matrix"""
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(docs)
        return cosine_similarity(tfidf_matrix)
    
    def plot_keyword_comparison(self, doc1_freq, doc2_freq, title="Keyword Frequency Comparison"):
        """Plot keyword frequencies comparison"""
        # Combine all keywords
        all_keywords = set(doc1_freq.keys()) | set(doc2_freq.keys())
        
        # Create lists for plotting
        keywords = list(all_keywords)
        doc1_values = [doc1_freq.get(k, 0) for k in keywords]
        doc2_values = [doc2_freq.get(k, 0) for k in keywords]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Keywords': keywords,
            'Document 1': doc1_values,
            'Document 2': doc2_values
        })
        
        # Melt DataFrame for seaborn
        df_melted = df.melt(id_vars=['Keywords'], 
                           var_name='Document', 
                           value_name='Frequency')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Keywords', y='Frequency', hue='Document')
        plt.xticks(rotation=45)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def analyze_documents(self, doc1, doc2, top_n=10):
        """Perform complete document analysis"""
        # Preprocess documents
        doc1_processed = self.preprocess_text(doc1)
        doc2_processed = self.preprocess_text(doc2)
        
        # Get keyword frequencies
        doc1_freq = self.get_keyword_frequencies(doc1, top_n)
        doc2_freq = self.get_keyword_frequencies(doc2, top_n)
        
        # Calculate similarity
        similarity = self.calculate_tfidf_similarity([doc1_processed, doc2_processed])[0][1]
        
        # Plot comparison
        self.plot_keyword_comparison(doc1_freq, doc2_freq)
        
        return {
            'similarity_score': similarity,
            'doc1_top_keywords': doc1_freq,
            'doc2_top_keywords': doc2_freq
        }

# Example usage
comparator = DocumentComparator()

# Example documents
doc1 = """
Machine learning is a subset of artificial intelligence that focuses on developing systems 
that can learn and improve from experience without being explicitly programmed. It uses 
statistical techniques to enable computers to learn from data.
"""

doc2 = """
Artificial intelligence encompasses machine learning techniques where computers can be 
trained to analyze data and make decisions. These systems use statistical approaches 
to learn patterns from experience.
"""

# Analyze documents
results = comparator.analyze_documents(doc1, doc2)

print(f"Similarity Score: {results['similarity_score']:.2f}\n")
print("Top Keywords Document 1:")
for word, freq in results['doc1_top_keywords'].items():
    print(f"{word}: {freq}")
print("\nTop Keywords Document 2:")
for word, freq in results['doc2_top_keywords'].items():
    print(f"{word}: {freq}")

# Additional Analysis: Word Clouds
from wordcloud import WordCloud

def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

create_wordcloud(doc1, 'Document 1 Word Cloud')
create_wordcloud(doc2, 'Document 2 Word Cloud')