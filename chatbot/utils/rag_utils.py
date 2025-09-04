import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def save_vectorizer_state(vectorizer: TfidfVectorizer, filename: str) -> None:
    """
    Save the state of a TfidfVectorizer to a file.
    
    Parameters:
    - vectorizer: The TfidfVectorizer instance to save.
    - filename: The name of the file to save the vectorizer state to.
    """
    # Before saving, store the key attributes needed for proper restoration
    vectorizer_data = {
        'vocabulary_': vectorizer.vocabulary_,
        'idf_': vectorizer.idf_,
        '_tfidf': vectorizer._tfidf,
        'params': vectorizer.get_params(),
    }

    # Save both the vectorizer and its data
    with open(filename, 'wb') as f:
        pickle.dump((vectorizer, vectorizer_data), f)


def load_vectorizer_state(vectorizer: TfidfVectorizer, vectorizer_data: dict) -> TfidfVectorizer:
    # Restore the internal attributes
    vectorizer.set_params(**vectorizer_data['params'])
    vectorizer.vocabulary_ = vectorizer_data['vocabulary_']
    vectorizer.idf_ = vectorizer_data['idf_']
    vectorizer._tfidf = vectorizer_data['_tfidf']

    return vectorizer


def load_vectorizer(filename: str) -> TfidfVectorizer:
    """
    Load the a TfidfVectorizer from a file.
    
    Parameters:
    - filename: The name of the file to load the vectorizer state from.
    """
    with open(filename, 'rb') as f:
        vectorizer, vectorizer_data = pickle.load(f)
        
    # Restore the internal attributes
    vectorizer = load_vectorizer_state(vectorizer, vectorizer_data)
    
    return vectorizer


def extract_domain_from_url(url: str) -> str:
    """
    Extract the domain name from the URL. May begin with http:// or https:// or even without protocol. 
    """
    # Remove protocol and www if present
    url = re.sub(r'^https?://(www\.)?', '', url)
    # Remove everything after the domain name and the trailing slash if present
    url = url.split('/')[0]
    return url


def format_source(source: str) -> str:
    """
    Format the source string to be lowercase and hyphenated.
    """
    # Convert to lowercase and replace non-alphanumeric characters with hyphens
    formatted_source = re.sub(r'\W+', '-', source.lower())
    formatted_source = formatted_source.strip('-')  # Remove leading/trailing hyphens
    return formatted_source


def get_context_source(metadata: dict) -> str:
    """
    Format the source of a context document.
    """
    source = metadata.get("source")
    if source is None:
        url = metadata.get("url")
        if url:
            source = extract_domain_from_url(url)
        else:
            source = "Unknown"

    # Format source as lowercase and hyphenated
    source = format_source(source)
    return source
