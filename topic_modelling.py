# CLEAN AND READY TO BE PUSHED TO GIT

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


def remove_urls(text):
    """remove hypertext links"""
    text = re.sub(r'https : ', 'https:', text)  # to account for NOW format (e.g., https : //www.brcgs.com/)
    text = re.sub(r'http : ', 'http:', text)
    text = re.sub(r'http\S+', '', text)  # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-
    # a-string-in-python/49257661 using tolgayilmaz solution
    text = re.sub(r'www\S+', '', text)
    return text

def remove_html_tags(text):
    """remove html tags"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_punct(text):
    """remove some common punctuation including full stops, excluding '-' to keep hyphenated compounds. But, removing
    ' -' when preceded by whitespace (e.g., ' -year-old', which gets created by removing digits).
    Note that ''' can be removed at this stage as Vader contains no contractions bar 'can't stand' which gets removed
    anyway because it has ' ' in it."""
    punctuations = '''!()[]{};:"'\,<>./?@#$%^&*_©~'''
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")
    text = re.sub(" -", " ", text)
    text = re.sub("----", "", text)
    return text

def retain_only_nouns(text):  # added 04022022
    """Use nltk to retain all different variants of nouns but nothing else"""
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    text = [w for w, p in tags if (p == 'NN' or p == 'NNP' or p == 'NNS' or p == 'NNPS')]
    text = ' '.join(text)
    return text

def apply_funs_and_lambdas(mdd):
    """Apply both functions defined above and lambdas, to lower, remove digits, remove 1-letter words,
    remove double spaces, retain only nouns (added 04022022), and strip. wamrp is for whole_art_minus_removes,
     processed."""
    mdd['wamrp'] = mdd['whole_art_minus_removes'].map(str) \
        .map(remove_urls) \
        .map(remove_html_tags) \
        .map(lambda x: x.lower()) \
        .map(lambda x: re.sub(r'\d+', '', x)) \
        .map(remove_punct) \
        .map(lambda x: re.sub(r'\b\w{1,1}\b', '', x)) \
        .map(lambda x: re.sub(' +', ' ', x)) \
        .map(retain_only_nouns) \
        .map(lambda x: x.strip())
    return mdd[['text_id', 'wamrp']]

def show_topics(vectorizer, lda_model, n_words=20):
    """function to investigate topic model output"""
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


# Create stop words list comprising Vader and other words
vader_words_list = list(analyzer.lexicon.keys())
stop_words = [v for v in vader_words_list if " " not in v and 'Þ' not in v]  # you need to ignore vader words with
# spaces in them. For 1, you're mean_vader_valence value was applied to individual words split by whitespace. For 2,
# your TF-IDF vectoriser also currently splits on whitespace (and only whitespace, following your specified
# token_pattern).
# Words containing Thorn letter also removed (which is in 2 vader-labelled emojis), as they otherwise
# generate warning during tf-idf vectorisation. Not interested in these in art text.
additional_stops = ['hit', 'shot', '-', '--', 've', 'nt', 'th', 'ca']  # remove '-' to get rid of hyphen between
# different words (e.g. 'some text - other text'), as these kept in art text as collateral from retention of hyphenated
# compounds. Also remove nonesense words (e.g. 'nt'). Also remove 2 words not in Vader that are very likely negative
# and not indicative of a specific news topic ('hit' and 'shot').
stop_words = stop_words + additional_stops
stop_words = stop_words + list(text.ENGLISH_STOP_WORDS)  # adding normal stopwords
stop_words = list(set(stop_words))

# By working with mean_valence_vader_words, you're kind of assuming that the Vader corpus contains all neg and pos
# words. Still, this assumption being wrong I think just means that the mean_valence_vader_words has a bit of extra
# variance. Conversely, going with prop_neg_words gets biased towards zero whenever a neg word is not in the
# Vader corpus.

# train test split not carried out as not prioritising prediction


# Create a tokenizer. This is not a lemmatizer , as Vader words aren't stemmed/lemmatized
vectorizer = CountVectorizer(stop_words=stop_words,
                             tokenizer=None,
                             ngram_range=(1, 1),
                             max_df=0.85,
                             min_df=1,
                             max_features=2000,
                             token_pattern=r"(?u)\S\S+")


# Define search parameters
search_params = {'n_components': [3, 4, 5]}  # 6 topics does produce better log lik score, but topic content then makes
# far less sense


# Load datasets and apply fun.s
mdd_G = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mddat_guardian.csv')
mdd_NYT = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mddat_nytimes.csv')
mdd_NYP = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mddat_nypost.csv')
mdd_DM = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mddat_dailymail.csv')

df_list = [mdd_G, mdd_NYT, mdd_NYP, mdd_DM]
df_list = [df.pipe(apply_funs_and_lambdas) for df in df_list]
mdd_all = pd.concat(df_list, ignore_index=True)

dtm_vects = vectorizer.fit_transform(mdd_all['wamrp']).toarray()
lda = LatentDirichletAllocation()  # Init model
model = GridSearchCV(lda, param_grid=search_params, n_jobs=7)  # Init Grid Search Class
# log-likelihood will be score to determine best model

print("reached searching")
model.fit(dtm_vects)
best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)  # Model Parameters
print("Best Log Likelihood Score: ", model.best_score_)  # Log Likelihood Score
print("Model Perplexity: ", best_lda_model.perplexity(dtm_vects))  # Perplexity
model.cv_results_['mean_test_score']  # to look at log lik for all no.s of topics
# array([-18474938.0687998 , -18306521.108075045, -18240422.06283749])  # for 4, 5, 6 topics

lda_output = best_lda_model.transform(dtm_vects)
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames)
dominant_topic = np.argmax(df_document_topic.values, axis=1)  # Get dominant topic for each document
df_document_topic['dominant_topic'] = dominant_topic
topic_columns = [col for col in df_document_topic.columns if col.startswith('Topic')]
df_document_topic['how_dom'] = np.max(df_document_topic[topic_columns].values, axis=1)  # To show how dominant pred is

df_document_topic['text_id'] = mdd_all['text_id']

df_document_topic['paper'] = 'G'
df_document_topic.loc[df_document_topic['text_id'].isin(mdd_DM['text_id']), 'paper'] = 'DM'
df_document_topic.loc[df_document_topic['text_id'].isin(mdd_NYP['text_id']), 'paper'] = 'NYP'
df_document_topic.loc[df_document_topic['text_id'].isin(mdd_NYT['text_id']), 'paper'] = 'NYT'

df_document_topic.groupby('dominant_topic').size()
df_document_topic.groupby(['dominant_topic', 'paper']).size()


# Examine topics
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=50)
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]


# Save df.s to csv
df_document_topic.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/art_topic.csv', index=False)
df_topic_keywords.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/topic_keywords.csv')
