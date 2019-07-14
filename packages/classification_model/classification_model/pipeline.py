from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from classification_model.processing import preprocessors as pp
from classification_model.config import config

PIPELINE_NAME = 'classification'

full_pipe = Pipeline(
    [
        ('preprocessor', pp.NLP_Preprocessor(variables = config.FEATURES)),
        # ('count', CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf', TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')),
        # ('tfidf_ngrams', TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3))),    
        # ('tfidf_chars', TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3))),
        ('gbc', GradientBoostingClassifier(random_state=config.SEED))
    ])
