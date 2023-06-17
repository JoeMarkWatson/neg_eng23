# READY TO BE PUSHED TO GIT

import pandas as pd
import re
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from numpy import arange



removes = ['<h', '<p', '\n', "' <h", "' <p", 'Advertisement <p', 'Advertisement <h', 'Share <p',
           'Share <h']  # for removing from sentences before applying Vader

tcm = ['RELATED ARTICLES <', 'Share this article <', 'Your details from Facebook will',
       'like it to be posted to Facebook', 'marketing and ads in line with our',
       'confirm this for your first post to Facebook', 'link your MailOnline account with',
       'will automatically post your comment and', 'time it is posted on MailOnline', 'link your MailOnline account',
       'first post to Facebook', 'comment will be posted to MailOnline', 'automatically post your MailOnline',
       'Share or comment on this']  # to remove 'share on facebook'-type requests - sometimes interspersed
       # with "@ @" symbols - without removing actual article sentences

appp = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/all_polit_party_people.csv')
appp = appp[appp['rank'] <= 100]
c_strings = list(appp[appp['group_name'] == 'Conservative']['person'])
l_strings = list(appp[appp['group_name'] == 'Labour']['person'])
d_strings = list(appp[appp['group_name'] == 'Democrat']['person'])
r_strings = list(appp[appp['group_name'] == 'Republican']['person'])

lib_id_strings = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/LiberalIdentity.txt', header=None)
con_id_strings = pd.read_csv('/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/ConservativeIdentity.txt', header=None)
con_id_strings = con_id_strings[con_id_strings[0] != 'fascist*']  # to remove sole Vader (valenced) word, which would
# otherwise impede later neg vs out/in-group interaction modelling
lidss = [l for l in lib_id_strings[0] if " " not in l]  # lib_id_strings single
cidss = [c for c in con_id_strings[0] if " " not in c]    # con_id_strings single
lidsm = [" " + l + " " for l in lib_id_strings[0] if l not in lidss]  # lib_id_strings multiple. Space added to avoid wildcard matching, given later code
cidsm = [" " + c + " " for c in con_id_strings[0] if c not in cidss]    # con_id_strings multiple
idth = [re.sub("\\*", '', t) for t in lidss + cidss if '-' in t]  # hyphenated words, used as conditional in get_art_desc_senti
lidss = [re.escape(l).replace('\*', '.*') for l in lidss]
cidss = [re.escape(c).replace('\*', '.*') for c in cidss]

def get_source_info(year):
    """returns df containing source files info, including url and title"""
    d = []
    source_files = os.listdir(root_path + "NOW" + str(year) + "n/source_files")
    for f in source_files:
        if f != '.DS_Store':
            print(f)
            with open(root_path + "NOW" + str(year) + "n/source_files/" + f, "r", encoding='cp1252',
                      errors="replace") as file:
                for line in file:
                    fields = line.split("\t")
                    if len(fields) > 4:
                        if fields[0].isdigit():
                            if paper_string in fields[5]:
                                d.append(
                                    {
                                        'text_id': fields[0],
                                        'year': year,
                                        'art_word_len': fields[1],
                                        'date': fields[2],
                                        'country': fields[3],
                                        'website': fields[4],
                                        'url': fields[5],
                                        'title': fields[6],
                                    }
                                )
    return pd.DataFrame(d)

def get_art_desc_senti(year, dd):
    """returns df containing art true word length and sentiment info"""
    year_dd = dd[dd['year'] == year]  # to reduce searching in 'if fields[0] in list(year_dd['text_id']):'
    d = []
    text_file_name = 'text_files' if paper_string in ['www.dailymail.co.uk', 'www.theguardian.com/'] else 'text_filesUS'
    list_loc = root_path + "NOW" + str(year) + "n/" + text_file_name
    text_files = os.listdir(list_loc)
    for f in text_files:
        if f != '.DS_Store':
            print(f)
            with open(root_path + "NOW" + str(year) + "n/" + text_file_name + "/" + f, "r", encoding='cp1252',
                      errors="replace") as file:  # fine to replace with '?' as raw NOW data uses this character in
                # place of characters such as '£'
                for line in file:
                    fields = line.split(" ", 1)  # maxsplit of 1
                    if len(fields) > 1:
                        fields[0] = re.sub("[^0-9]", "", fields[0])  # retain only numbers from fields[0]
                        if len(fields[0]) > 1:  # i.e., not NA and therefore .isdigit()
                            if fields[0] in list(year_dd['text_id']):
                                sentences = re.split("\. |\> |! ", fields[1])  # '?' not used given with open comment
                                sentences = [s for s in sentences if
                                             s not in removes and len(s) > 1 and not any(t in s for t in tcm)]
                                art_sentences_len = len(sentences)
                                if len(sentences) > 1:  # added for 2019 DM data to catch art with no text

                                    wamr = '. '.join(sentences)  # whole_art_minus_removes

                                    c_count = len(re.findall('|'.join(c_strings), wamr))  # UK Con count
                                    r_count = len(re.findall('|'.join(r_strings), wamr))  # US Rep count
                                    d_count = len(re.findall('|'.join(d_strings), wamr))  # US Dem count
                                    l_count = len(re.findall('|'.join(l_strings), wamr))  # UK Lab count
                                    # Note - art text formatting means above does find a possessive name reference and
                                    # an end-of-sentence name reference

                                    # Case ignored in the following 2 searches, to correspond with later cons/lib term
                                    # searches among individ words transformed to lower
                                    US_lib_id_count = len(re.findall(lidsm[0], wamr, flags=re.IGNORECASE))  # lidsm term count, of which 1 term only
                                    US_con_id_count = len(re.findall(cidsm[0], wamr, flags=re.IGNORECASE))  # cidsm term count, of which 1

                                    my_word_count, my_word_len_sum, neg_my_word_count, pos_my_word_count,  total_valence, \
                                        art_vs_neg_comp_count, art_vs_clear_neg_comp_count = 0, 0, 0, 0, 0, 0, 0

                                    for s in sentences:
                                        words_s = nltk.word_tokenize(s)
                                        words_s = [w.lower() for w in words_s]
                                        for ws in words_s:
                                            if len(re.findall(r'\w+', ws)) == 1 or any(ws.startswith(term) for term in idth):
                                                my_word_count += 1
                                                my_word_len_sum += len(ws)
                                                neg_my_word_count += analyzer.polarity_scores(ws)['neg']
                                                pos_my_word_count += analyzer.polarity_scores(ws)['pos']
                                                total_valence += analyzer.polarity_scores(ws)['compound']  # Compound is
                                                # considered OK (as opposed to using individual user ratings). Using it promotes
                                                # ease of replication. Vader's documentation says it is normalised to be between -1 and 1,but the
                                                # most extreme neg and pos values are, respectively, 'rape' at -0.71 and 'magnificently' at 0.66.
                                                # I would also wager that the Vader dict on its own is skewed towards negative valence.

                                                for l_term in lidss:
                                                    if re.fullmatch(l_term, ws):
                                                        US_lib_id_count += 1
                                                for c_term in cidss:
                                                    if re.fullmatch(c_term, ws):
                                                        US_con_id_count += 1

                                        vs = analyzer.polarity_scores(s)
                                        if vs['compound'] < 0:  # for prop of negative, neutral and positive sentences
                                            art_vs_neg_comp_count += 1
                                        if vs['compound'] <= -0.05:  # true vals follow https://github.com/cjhutto/vaderSentiment
                                            art_vs_clear_neg_comp_count += 1
                                    mean_word_len = my_word_len_sum / my_word_count
                                    mean_valence_vader_words = total_valence / (neg_my_word_count + pos_my_word_count) if \
                                        (neg_my_word_count + pos_my_word_count) > 0 else "NA"
                                    d.append(
                                        {
                                            'text_id': fields[0],
                                            'art_sentences_len': art_sentences_len,
                                            'my_word_count': my_word_count,
                                            'vader_words_count': neg_my_word_count + pos_my_word_count,
                                            'art_neg_word_prop': neg_my_word_count / my_word_count,
                                            'art_pos_word_prop': pos_my_word_count / my_word_count,  # prop_neg_minus_prop_pos can be added at any later point
                                            'mean_valence_all_words': total_valence / my_word_count,  # this will be biased toward zero when author vocab does not overlap much with Vader words
                                            # note that non-Vader words have a compound of 0 (and neutral of 1)
                                            'mean_valence_vader_words': mean_valence_vader_words,  # likely your best IV
                                            'prop_of_words_vader_words': (neg_my_word_count + pos_my_word_count) / my_word_count,  # good descriptive info and potentially useful for sense checking
                                            'mean_my_word_len': mean_word_len,
                                            'mean_my_words_in_sen': my_word_len_sum / art_sentences_len,
                                            'art_neg_sent_prop': art_vs_neg_comp_count / art_sentences_len,
                                            'art_true_neg_sent_prop': art_vs_clear_neg_comp_count / art_sentences_len,
                                            'UK_cons_count': c_count,
                                            'UK_lab_count': l_count,
                                            'US_rep_count': r_count,
                                            'US_dem_count': d_count,
                                            'US_lib_id_count': US_lib_id_count,
                                            'US_con_id_count': US_con_id_count,
                                            'whole_art_minus_removes': wamr
                                        }
                                    )
    return pd.DataFrame(d)


def exclude_non_arts(mdd):

    if paper_string == 'www.nytimes.com':
        substrings_to_exclude = ['aponline', 'reuters', 'realestate', 'archive', '/podcasts/', 'popcast', '-podcast-',
                                 '/video/', 't-magazine', '/magazine/', '/column/', '/crosswords/', '/at-home',
                                 '/subscription/', '/multiproduct/', 'the-daily', '/briefing/', '/letters/', 'book-list',
                                 '/by/', '/live/']  # removes AP links, realestate art.s (predom listings), 'archive info,
        # audio files, mag content, columns containining links to multiple articles, puzzles/crosswords,
        # subscription sign-up info, daily briefings, book lists, and live event updates
        paper_names = ['New York Times', 'nytimes.com', 'The New York Times', 'www.nytimes.com']
        mdd = mdd[mdd['website'].isin(paper_names)]  # remove those listed as other websites (inc 'Sun Sentinel')

    if paper_string == 'www.dailymail.co.uk':
        substrings_to_exclude = ['/galleries/', '/wires/', '/search.html', '/index',
                                 'https://www.dailymail.co.uk/home/index.rss%20']  # remove url links that are assoc
        # press, site-wide keyword searches, links that show all trending news stories about a topic, an erroneous URL
        mdd = mdd[~((mdd['url'].str.contains('books/article') == True) & (mdd['url'].str.contains('/amp/') == True))]
        # remove url links to blurbs of one or multiple books in a book category, presented without additional comment

    if paper_string == 'www.washingtonpost.com':
        substrings_to_exclude = ['letters-to-the-ed', 'family-letters', '/podcasts/', '-podcast-',
                                 '/ap-', '_quiz.html', '/people/', 'washington-post-live',
                                 'live-updates', '-live/', '-video-', '/video/', '/archive?']
        # if you want to remove all AP articles, you need to use article text as not reg reflected in URL.
        # E.g., "https://www.washingtonpost.com/politics/2023/04/10/macron-taiwan-china-eu-us-interview/dab0103e-d7af-11ed-aebd-3fd2ac4c460a_story.html"

    if paper_string == 'www.theguardian.com/':
        substrings_to_exclude = list(arange(2001, 2019))  # article URLs signifying pre 2019
        substrings_to_exclude = ['/' + str(ste) + '/' for ste in substrings_to_exclude]
        substrings_to_exclude = substrings_to_exclude + ['discussion/p', '/profile/', '/rss', '/help/']  # only article comments, lists of articles
        # authored by a specific journo, URLs containing '/rss' are XML versions of summary pages containing many
        # stories, and help pages, e.g., '/help/contact-us'

        # mdds = mdds[~mdds['url'].str.contains('/live/')]  # are live event updates e.g.:
        # https://www.theguardian.com/football/live/2019/feb/02/chelsea-v-huddersfield-everton-v-wolves-and-more-football-live
        # so could consider excluding as not classic articles

    if paper_string == 'nypost.com/':
        substrings_to_exclude = ['share=']
        substrings_to_retain = ['https://nypost.com/2019/', 'https://nypost.com/2020/', 'https://nypost.com/2021/']
        mdd = mdd[mdd['url'].str.contains('|'.join(substrings_to_retain))]
        mdd = mdd[mdd['website'] == 'New York Post']  # keep only art.s where 'New York Post' is website
        mdd = mdd[mdd['title'].str.contains('nypost.com') == False]  # remove any rows with erroneous title info
        # (signified by the presence of 'nypost.com')

    mdd = mdd[~mdd['url'].str.contains('|'.join(substrings_to_exclude))]

    return mdd


def get_title_desc_senti(title):
    """takes title and produces list of senti info"""

    d = []
    my_word_count, my_word_len_sum, neg_my_word_count, pos_my_word_count, total_valence = 0, 0, 0, 0, 0

    manip_title = re.sub('<strong>|</strong>|\|''|\.\.\.\n', '', title)
    manip_title = re.sub(r'\s+', ' ', manip_title).strip()

    words_s = nltk.word_tokenize(manip_title)
    words_s = [w.lower() for w in words_s]
    for ws in words_s:
        if len(re.findall(r'\w+', ws)) == 1:  # so only counting words where at least one letter
            my_word_count += 1
            my_word_len_sum += len(ws)
            neg_my_word_count += analyzer.polarity_scores(ws)['neg']
            pos_my_word_count += analyzer.polarity_scores(ws)['pos']
            total_valence += analyzer.polarity_scores(ws)['compound']  # Compound is
            # considered OK (as opposed to using individual user ratings). Using it promotes
            # ease of replication. Vader's documentation says it is normalised to be between -1 and 1. The
            # most extreme neg and pos values are, respectively, 'rape' at -0.71 and 'magnificently' at 0.66.
            # I would wager that the Vader dict on its own is skewed towards negative valence.
    vs = analyzer.polarity_scores(manip_title)
    neg_title = 1 if vs['compound'] < 0 else 0
    clear_neg_title = 1 if vs['compound'] <= -0.05 else 0  # true vals follow https://github.com/cjhutto/vaderSentiment

    mean_word_len = my_word_len_sum / my_word_count if my_word_count > 0 else "NA"
    mean_valence_vader_words = total_valence / (neg_my_word_count + pos_my_word_count) if \
        (neg_my_word_count + pos_my_word_count) > 0 else "NA"
    d.append(
        {
            'manip_title': manip_title,
            'my_word_count_title': my_word_count,
            'vader_words_count_title': neg_my_word_count + pos_my_word_count,
            'neg_word_prop_title': neg_my_word_count / my_word_count if my_word_count > 0 else "NA",
            'title_pos_word_prop_title': pos_my_word_count / my_word_count if my_word_count > 0 else "NA",
            # prop_neg_minus_prop_pos can be added at any later point
            'mean_valence_all_words_title': total_valence / my_word_count if my_word_count > 0 else "NA",
            # this will be biased toward zero when author vocab does not overlap much with Vader words
            # note that non-Vader words have a compound of 0 (and neutral of 1)
            'mean_valence_vader_words_title': mean_valence_vader_words,  # likely your best IV
            'prop_of_words_vader_words_title': (neg_my_word_count + pos_my_word_count) / my_word_count if my_word_count > 0 else "NA",
            # good descriptive info and potentially useful for sense checking
            'mean_my_word_len_title': mean_word_len,
            'all_title_compound': vs['compound'],
            'all_title_neg': neg_title,
            'all_title_clear_neg': clear_neg_title,
        }
    )
    return d


def save_mdd(mdd):
    """saves mdd plus article text - mddat"""
    if paper_string == 'nypost.com/':
        paper = 'nypost'
    else:
        paper = re.split('\.', paper_string)[1]
        paper = re.sub('the', '', paper)
    save_loc = '/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/mddatcf_' + paper + '.csv'
    mdd.to_csv(save_loc, index=False)



# # # #

root_path = "/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/NOWn/"

paper_string = 'www.dailymail.co.uk'
paper_string = 'www.theguardian.com/'
paper_string = 'www.nytimes.com'
paper_string = 'nypost.com/'
#paper_string = 'www.washingtonpost.com'  # trialling suggests rel few art.s are tweeted about

# extract source file info
dd_2019 = get_source_info(2019)
dd_2020 = get_source_info(2020)
dd_2021 = get_source_info(2021)
dd_all = pd.concat([dd_2019, dd_2020, dd_2021])

# extract text file info
dd2_2019 = get_art_desc_senti(2019, dd=dd_all)
dd2_2020 = get_art_desc_senti(2020, dd=dd_all)
dd2_2021 = get_art_desc_senti(2021, dd=dd_all)
dd2_all = pd.concat([dd2_2019, dd2_2020, dd2_2021])

# inner join dd_all and dd2_all into mdd
mdd = pd.merge(dd_all, dd2_all, on='text_id')  # dd2 only attempted senti for art.s in dd, so len(dd) == len(dd2)
mdd = mdd.drop_duplicates(subset='url', keep="first")  # remove any url duplicates
mdd['text_id'] = mdd['text_id'].astype('int64')

# remove non-articles
mdd = exclude_non_arts(mdd)
mdd = mdd.reset_index(drop=True)

ds = list(map(lambda x: get_title_desc_senti(x)[0], mdd['title']))
ds = pd.DataFrame(ds)
mdd = pd.concat([mdd, ds], axis=1)

save_mdd(mdd)


# # # Investigating title text clipping within mdd.s

# Some prop of titles in each mdd file is clipped. Clipping may be due in part to title length, but inspection of
# DM titles shows this not to be a consistent pattern.

#non_clipped_titles = [m for m in mdd['title'] if '...\n' not in m]
#1-len(non_clipped_titles)/len(mdd)  # prop of titles clipped

# 0.0339 for G, 0.9459 for DM, 0.0046 for NYP, 0.12137 for NYT. So you still have thousands of complete art titles to
# work with for each indiv paper, bar DM (where only 860 are non-clipped)
