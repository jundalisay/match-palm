import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.graph_objects as go
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')



nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')



class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
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
    
    def get_keyword_frequencies(self, text):
        """Get frequency distribution of keywords"""
        tokens = word_tokenize(self.preprocess_text(text))
        freq_dist = nltk.FreqDist(tokens)
        return dict(freq_dist)
    
    def calculate_similarity(self, text1, text2):
        """Calculate TF-IDF similarity between texts"""
        tfidf = TfidfVectorizer()
        texts = [self.preprocess_text(text1), self.preprocess_text(text2)]
        tfidf_matrix = tfidf.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)[0][1]

def main():
    st.set_page_config(page_title="Job Match Analyzer", layout="wide")
    
    st.title("Job Match Keyword Analyzer")
    st.markdown("""
    Compare job requirements with applicant qualifications to see keyword similarity.
    Enter text in both fields (max 10,000 characters each).
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        employer_text = st.text_area(
            "Employer Requirements",
            height=1000,
            max_chars=10000,
            placeholder="Enter job requirements...(max 1,000 characters)"
        )
        
    with col2:
        applicant_text = st.text_area(
            "Applicant Qualifications",
            height=1000,
            max_chars=10000,
            placeholder="Enter qualifications...(max 1,000 characters)"
        )
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    if employer_text and applicant_text:
        st.markdown("---")
        
        # Calculate similarity score
        similarity = analyzer.calculate_similarity(employer_text, applicant_text)
        
        # Get keyword frequencies
        employer_freq = analyzer.get_keyword_frequencies(employer_text)
        applicant_freq = analyzer.get_keyword_frequencies(applicant_text)
        
        # Create combined keyword set
        all_keywords = set(employer_freq.keys()) | set(applicant_freq.keys())
        
        # Create DataFrame for comparison
        df = pd.DataFrame({
            'Keyword': list(all_keywords),
            'Employer': [employer_freq.get(k, 0) for k in all_keywords],
            'Applicant': [applicant_freq.get(k, 0) for k in all_keywords]
        })
        
        # Sort by total frequency
        df['Total'] = df['Employer'] + df['Applicant']
        df = df.sort_values('Total', ascending=False).head(10)
        
        # Display similarity score
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.metric(
                "Match Score",
                f"{similarity:.0%}",
                help="Similarity score based on keyword frequency and importance"
            )
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(name='Employer', x=df['Keyword'], y=df['Employer'], marker_color='#1f77b4'),
            go.Bar(name='Applicant', x=df['Keyword'], y=df['Applicant'], marker_color='#ff7f0e')
        ])
        
        fig.update_layout(
            title='Top 10 Keyword Comparison',
            xaxis_title='Keywords',
            yaxis_title='Frequency',
            barmode='group',
            height=500,
            margin=dict(t=30)
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display keyword table
        st.markdown("### Detailed Keyword Analysis")
        st.dataframe(
            df[['Keyword', 'Employer', 'Applicant']].style.format({
                'Employer': '{:.0f}',
                'Applicant': '{:.0f}'
            }),
            hide_index=True
        )

if __name__ == "__main__":
    main()