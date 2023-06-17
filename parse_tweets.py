import os
import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


root_path = '/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files'

def create_text_id_dict(paper_name):
    df_loc = root_path + '/mdd_' + paper_name + '.csv'
    mdd = pd.read_csv(df_loc)
    text_id_dict = dict(zip(mdd['url'], mdd['text_id']))
    return text_id_dict


def reset_values():
    return np.repeat(0, 4)


def digitise(tns):
    """Digitise scraped tweet number strings"""
    if tns:
        if 'K' in tns:
            int_tns = float(tns.replace('K', ''))
            int_tns = int(int_tns*1000)
        else:
            int_tns = int(tns.replace(',', ''))
    else:
        int_tns = 0
    return int_tns


def parse_tweets_agg(folders, text_id_dict):
    """# UR DOING SOMETHING THAT AT LEAST SHOWS U WHAT FAILED TO BE SCRAPED, AND WHAT GOT NO SCRAPES.
    JUST WORK WITH G FOR NOW. DO NYP AFTER (WITH THAT REQ.ING U TO CREATE A JOINT FOLDER)."""

    d = []

    for folder in folders:

        files_loc = root_path + '/' + folder
        indiv_files = os.listdir(files_loc)
        for indiv in indiv_files:
            if indiv != ".DS_Store":
                n_tweets, n_replies, n_RTs, n_likes = reset_values()
                with open(files_loc + '/' + indiv) as file:
                    for k, line in enumerate(file):
                        line_fields = line.split('§')
                        if line_fields[1] != 'NT':  # is a date unless no tweets, in which case 'NT'
                            n_tweets += 1
                            n_replies += digitise(line_fields[5])
                            n_RTs += digitise(line_fields[6])
                            n_likes += digitise(line_fields[7])
                    d.append({
                        'text_id': text_id_dict[line_fields[0]],  # id of newspaper article text, obtained using dict
                        'group': folder,
                        'n_tweets': n_tweets,
                        'n_replies': n_replies,
                        'n_RTs': n_RTs,
                        'n_likes': n_likes,
                    })
    return pd.DataFrame(d)


def parse_tweets_indiv(folders, text_id_dict):
    """parses all tweets individually"""

    d = []

    for folder in folders:

        files_loc = root_path + '/' + folder
        indiv_files = os.listdir(files_loc)
        for indiv in indiv_files:

            if indiv != ".DS_Store":
                with open(files_loc + '/' + indiv) as file:
                    for line in file:

                        my_word_count, my_word_len_sum, neg_my_word_count, pos_my_word_count, total_valence = 0, 0, 0, \
                            0, 0

                        line_fields = line.split('§')
                        if line_fields[1] != 'NT':  # is a date unless no tweets, in which case 'NT'
                            tweet_text = line_fields[8]
                            tweet_text = re.sub('TWEET_CONTAINS_LINK_ONLY', '', tweet_text)  # pattern added by me when no text
                            tweet_text = re.sub(' NEW LINE ', ' ', tweet_text)  # pattern added by me when new line in tweet
                            tweet_text = re.sub('http:\/\/\S+|https:\/\/\S+|www\.\S+|www[2-9]\.\S+', '', tweet_text)  # remove links
                            tweet_length = len(tweet_text)

                            words_s = nltk.word_tokenize(tweet_text)
                            words_s = [w.lower() for w in words_s]
                            for ws in words_s:
                                if len(re.findall(r'\w+', ws)) == 1:  # so only counting words where at least one letter
                                    my_word_count += 1
                                    my_word_len_sum += len(ws)
                                    neg_my_word_count += analyzer.polarity_scores(ws)['neg']
                                    pos_my_word_count += analyzer.polarity_scores(ws)['pos']
                                    total_valence += analyzer.polarity_scores(ws)['compound']
                            vs = analyzer.polarity_scores(tweet_text)
                            neg_tweet = 1 if vs['compound'] < 0 else 0
                            clear_neg_tweet = 1 if vs['compound'] <= -0.05 else 0  # true vals follow https://github.com/cjhutto/vaderSentiment
                            mean_word_len = my_word_len_sum / my_word_count if my_word_count > 0 else "NA"
                            mean_valence_vader_words = total_valence / (neg_my_word_count + pos_my_word_count) if \
                                (neg_my_word_count + pos_my_word_count) > 0 else "NA"

                            d.append({
                                'text_id': text_id_dict[line_fields[0]],  # id of newspaper article text, obtained using dict
                                'twitter_handle': line_fields[3],
                                'tweet_id': line_fields[4],
                                'n_replies': digitise(line_fields[5]),
                                'n_RTs': digitise(line_fields[6]),
                                'n_likes': digitise(line_fields[7]),
                                'char_length_tweet': tweet_length,
                                'my_word_count_tweet': my_word_count,
                                'mean_my_word_len_tweet': mean_word_len,
                                'vader_words_count_tweet': neg_my_word_count + pos_my_word_count,
                                'neg_word_prop_tweet': neg_my_word_count / my_word_count if my_word_count > 0 else "NA",
                                'pos_word_prop_tweet': pos_my_word_count / my_word_count if my_word_count > 0 else "NA",
                                'mean_valence_all_words_tweet': total_valence / my_word_count if my_word_count > 0 else "NA",
                                'mean_valence_vader_words_tweet': mean_valence_vader_words,  # likely your best IV for individ tweets models
                                'prop_of_words_vader_words_tweet': (neg_my_word_count + pos_my_word_count) / my_word_count if my_word_count > 0 else "NA",
                                'all_tweet_compound': vs['compound'],
                                'all_tweet_neg': neg_tweet,
                                'all_tweet_clear_neg': clear_neg_tweet,
                                #'group': folder,
                                #'tweet_date': line_fields[1],
                                #'tweet_time': line_fields[2],
                                #'tweet_text': line_fields[8],
                            })

    return pd.DataFrame(d)


# Guardian folders
foldersG = ['Guardian_tweets', 'tweetsSelMissG']
foldersNYP = ['tweetsSelNYP_17', 'tweetsSelNYP11052023', 'tweetsSelMissNYP']
foldersDM = ['tweetsSelDM', 'tweetsSelDM12', 'tweetsSelDmMiss']
foldersNYT = ['tweetSelNYT20', 'tweetsSelNYTnMBP', 'tweetsSelNYTnMBP2a', 'tweetsSelNYTnMBP2b', 'tweetsSelNYTnMBP2c',
         'tweetsSelNYTnMBP2d', 'tweetsSelNYTnMBP3a', 'tweetsSelNYTnMBP3b', 'tweetsSelNYTnMBP3c', 'tweetsSelNYTnMBP3d',
         'tweetsSelNYTnMBP4a', 'tweetsSelNYTnMBP4b', 'tweetsSelNYTnMBP5a', 'tweetsSelNYTnMBP5b', 'tweetsSelNYTnMBP5c',
         'tweetsSelNYTnMBP5d', 'tweetsSelNYT', 'tweetsSelNYT2', 'tweetsSelNYTmissed1', 'tweetsSelNYTmissed2',
         'tweetsSelNYTmissed3', 'tweetsSelNYTmissed4', 'tweetsSelNYTmissed5', 'tweetsSelNYTmissed6',
         'tweetsSelNYTmissed7']


# GUARDIAN
# parse tweets on aggregate for each article
text_id_dict = create_text_id_dict('guardian')
tweets_aggG = parse_tweets_agg(folders=foldersG, text_id_dict=text_id_dict)
#tweets_aggG.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/aggG.csv', index=False)
#mdd_file = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mdd_guardian.csv')
#missing_searches = [t for t in list(mdd_file['text_id']) if t not in list(tweets_agg['text_id'])]  # ALL ORIGINALLY MISSING SEARCHES NOW COMPLETED
#missing_searches_df = mdd_file[mdd_file['text_id'].isin(missing_searches)]
#save_loc = '/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mdd_MissedG.csv'
#missing_searches_df.to_csv(save_loc, index=False)
# same process used for other files, with save files including: mdd_MissedNYP, mdd_MissedDM, mdd_MissedNYT25052023
tweets_dfG = parse_tweets_indiv(folders=foldersG, text_id_dict=text_id_dict)
#tweets_dfG.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/tweets_aggG.csv', index=False)

# NYP
# parse tweets on aggregate for each article
text_id_dict = create_text_id_dict('nypost')
tweets_aggNYP = parse_tweets_agg(folders=foldersNYP, text_id_dict=text_id_dict)
#tweets_aggNYP.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/aggNYP.csv', index=False)
tweets_dfNYP = parse_tweets_indiv(folders=foldersNYP, text_id_dict=text_id_dict)
#tweets_dfNYP.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/tweets_aggNYP.csv', index=False)

# DMO
# parse tweets on aggregate for each article
text_id_dict = create_text_id_dict('dailymail')
tweets_aggDM = parse_tweets_agg(folders=foldersDM, text_id_dict=text_id_dict)
#tweets_aggDM.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/aggDM.csv', index=False)
tweets_dfDM = parse_tweets_indiv(folders=['tweetsSelDM', 'tweetsSelDM12', 'tweetsSelDmMiss'], text_id_dict=text_id_dict)
#tweets_dfDM.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/tweets_aggDM.csv', index=False)

# NYT
text_id_dict = create_text_id_dict('nytimes')
tweets_aggNYT = parse_tweets_agg(folders=foldersNYT, text_id_dict=text_id_dict)
tweets_aggNYT = tweets_aggNYT.drop_duplicates(subset='text_id', keep="first")  # where duplicate then keep first, dropping 5 duplicate text_id's
#tweets_aggNYT.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/aggNYT.csv', index=False)
tweets_dfNYT = parse_tweets_indiv(folders=foldersNYT, text_id_dict=text_id_dict)
#tweets_dfNYT.to_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/tweets_aggNYT.csv', index=False)
