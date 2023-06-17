import pandas as pd
import datetime
import re
import pickle
import matplotlib.pylab as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import numpy as np
#from emoji import UNICODE_EMOJI
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.formula.api as smf
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from joblib import Parallel, delayed  # for parallel processing
from scipy.stats import shapiro
from scipy.stats import normaltest
from numpy import arange

# TODO: Add internal meta-analysis, in line with https://www.pnas.org/doi/full/10.1073/pnas.2024292118

# def functions

def add_dummies(df):
    """make dummies, store their col names and mix them into mddsl"""
    dummy_df = pd.get_dummies(
        df[['year', 'day_of_week', 'month']].astype(str))  # 'news_cat' removed, as likely strong overlap with art TF-IDF info
    dummy_df = dummy_df.drop(columns=['year_2019', 'day_of_week_0', 'month_01'])
    mddsld = pd.concat([df.reset_index(), dummy_df.reset_index()], axis=1, join='inner')
    return mddsld

def manip_df(df_mdd, df_tweets, topics):
    """manipulate df.s inc merging and creating vars employed in analysis"""

    df = pd.merge(
        df_mdd[['text_id', 'mean_valence_vader_words', 'my_word_count', 'mean_my_word_len', 'mean_my_words_in_sen',
                'vader_words_count', 'mean_valence_vader_words_title', 'date', 'UK_cons_count', 'UK_lab_count',
                'US_rep_count', 'US_dem_count', 'US_lib_id_count', 'US_con_id_count']], df_tweets)
    df = df[df['my_word_count'] > 99]
    # mdds = mdds[mdds['vader_words_count'] > 9]  # not adding this, as weakens comparison vs title neg

    df = pd.merge(df, topics[['text_id', 'dominant_topic']])  # add topic column
    df.dropna(subset=['mean_valence_vader_words'], how='all', inplace=True)

    df['year'] = list(map(lambda x: '20' + x[:2], df['date']))
    df['month'] = list(map(lambda x: x[3:5], df['date']))
    df['day_of_week'] = list(
        map(lambda x: datetime.date(int('20' + x[:2]), int(x[3:5]), int(x[-2:])).weekday(), df['date']))  # 0 is Mon

    df = add_dummies(df)

    df['my_word_count_log'] = np.log(df['my_word_count'] + 1)
    df['mean_my_word_len_log'] = np.log(df['mean_my_word_len'] + 1)
    df['mean_my_words_in_sen_log'] = np.log(df['mean_my_words_in_sen'] + 1)
    df['n_tweets_log'] = np.log(df['n_tweets'] + 1)

    df['n_tweets_log_CHECK'] = (np.log(df['n_tweets'] + 1))*2

    df['n_tweetsPlusRTs_log'] = np.log((df['n_tweets'] + df['n_RTs']) + 1)
    df['n_replies_log'] = np.log(df['n_replies'] + 1)
    df['n_RTs_log'] = np.log(df['n_RTs'] + 1)
    df['n_likes_log'] = np.log(df['n_likes'] + 1)

    df['any_pol_pers_term_US'] = df['US_rep_count'] + df['US_dem_count'] + df['US_lib_id_count'] + df['US_con_id_count'] > 0
    df['any_pol_pers_term_UK'] = df['UK_cons_count'] + df['UK_lab_count'] + df['US_lib_id_count'] + df['US_con_id_count'] > 0

    df['any_US_pol'] = df['US_rep_count'] + df['US_dem_count'] > 0
    df['any_UK_pol'] = df['UK_cons_count'] + df['UK_lab_count'] > 0

    df['US_rep_con_count'] = df['US_rep_count'] + df['US_con_id_count']
    df['US_dem_lib_count'] = df['US_dem_count'] + df['US_lib_id_count']
    df['UK_cons_con_count'] = df['UK_cons_count'] + df['US_con_id_count']
    df['UK_lab_lib_count'] = df['UK_lab_count'] + df['US_lib_id_count']

    df['maj_rep'] = df['US_rep_con_count'] > df['US_dem_lib_count']
    df['maj_dem'] = df['US_rep_con_count'] < df['US_dem_lib_count']
    df['maj_con'] = df['UK_cons_con_count'] > df['UK_lab_lib_count']
    df['maj_lab'] = df['UK_cons_con_count'] < df['UK_lab_lib_count']

    boolean_columns = ['any_pol_pers_term_US', 'any_pol_pers_term_UK', 'any_US_pol', 'maj_rep', 'maj_dem', 'any_UK_pol',
                       'maj_con', 'maj_lab']
    df[boolean_columns] = df[boolean_columns].astype(int)  # convert to facil output from later inf analysis funs

    df['neg_art'] = np.where(df['mean_valence_vader_words'] < 0, 1, 0)  # better for story and means that exact same neg
    # cutoff can be used for tweet-focused 'why' analysis
    df['neg_art_r'] = np.where(df['mean_valence_vader_words'] < df['mean_valence_vader_words'].mean(), 1,
                               0)  # mean-based neg_art cutoff used for robustness check

    df['neg_title'] = np.where(df['mean_valence_vader_words_title'] < 0, 1, 0)

    return df

def manip_i_df(df_i_tweets, manipd_df):
    df = pd.merge(df_i_tweets, manipd_df[['text_id', 'neg_art', 'neg_title', 'my_word_count_log', 'mean_my_word_len_log',
     'mean_my_words_in_sen_log', 'year_2020', 'year_2021', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3',
     'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12']], on='text_id')  # left/inner join

    df['char_length_tweet_log'] = np.log(df['char_length_tweet'])
    df['n_RTs_log'] = np.log(df['n_RTs'] + 1)
    df['n_replies_log'] = np.log(df['n_replies'] + 1)
    df['n_likes_log'] = np.log(df['n_likes'] + 1)

    return df

def print_plot_descs(df):
    """prints some mean values, makes some tweet dist plots, and runs an OLS model with no controls.
    You concede that the OLS coeff could have some positive bias, hence the use of controls in inf models. Maybe, for
    example, neg art.s are released on a high readership day. Or, perhaps certain topics that are more likely to be
    neg get shared more """

    print("Number of rows in df:", len(df))
    print("Proportion of news articles that are negative:", df['neg_art'].mean())
    print("Total number of tweets concerning all news articles:", df['n_tweets'].sum())
    print("Mean number of tweets per news article:", df['n_tweets'].mean())

    plt.hist(df['n_tweets'], bins=20)
    plt.show()  # v skewed when not standardised

    df['n_tweets_log'].mean()
    plt.hist(df['n_tweets_log'], bins=20)
    plt.show()

    # sum of all shares for all neg art.s or pos art.s
    # df[df['neg_art'] == 1]['n_tweets'].sum()
    # df[df['neg_art'] == 0]['n_tweets'].sum()  # could also be checked for with tweets + RTs, RTs only, replies and likes

    print(smf.ols("n_tweets_log ~ neg_art", data=df).fit().summary().tables[1])
    # same result produced by:
    # round(df[df['neg_art']==1]['n_tweets_log'].mean() - df[df['neg_art']==0]['n_tweets_log'].mean(), 4)

def run_standard_reg(Y, T, X, df, paper_name):
    """run a standard multivariate regression on df and print the output"""
    model_NT_1 = smf.ols(f"{Y}~{T}+{'+'.join(X)}", data=df).fit()
    print(model_NT_1.summary().tables[1])

    T_splits = T.split("+")

    point_estT1, lower_conf_intT1, upper_conf_intT1 = model_NT_1.params[T_splits[0]], \
        model_NT_1.conf_int(alpha=0.05).loc[T_splits[0]][0], \
        model_NT_1.conf_int(alpha=0.05).loc[T_splits[0]][1]

    if len(T_splits) > 1:
        point_estT2, lower_conf_intT2, upper_conf_intT2 = model_NT_1.params[T_splits[1]], \
            model_NT_1.conf_int(alpha=0.05).loc[T_splits[1]][0], \
            model_NT_1.conf_int(alpha=0.05).loc[T_splits[1]][1]

        if len(T_splits) == 2:
            point_estI, lower_conf_intI, upper_conf_intI = ['NA'] * 3

        else:
            point_estI, lower_conf_intI, upper_conf_intI = model_NT_1.params[re.sub("\\*", ":", T_splits[2])], \
                model_NT_1.conf_int(alpha=0.05).loc[re.sub("\\*", ":", T_splits[2])][0], \
                model_NT_1.conf_int(alpha=0.05).loc[re.sub("\\*", ":", T_splits[2])][1]

    else:
        point_estT2, lower_conf_intT2, upper_conf_intT2, \
            point_estI, lower_conf_intI, upper_conf_intI = ['NA'] * 6

    controls = 'core plus topic' if 'dominant_topic' in X else 'core'

    return [paper_name, 'multiple regression', Y, T, controls, len(df), point_estT1,
            lower_conf_intT1, upper_conf_intT1, point_estT2, lower_conf_intT2, upper_conf_intT2,
            point_estI, lower_conf_intI, upper_conf_intI]

def run_ps(df, X, T, Y):
    """compute the IPTW estimator within run_prop_score_model"""
    ps = LogisticRegression(penalty=None, max_iter=10000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]  # estimate the propensity score
    weight = (df[T] - ps) / (ps * (1 - ps))  # define the weights
    return np.mean(weight * df[Y])  # compute the ATE

def run_prop_score_model(df, X, T, Y, paper_name, bss=1000, ran_seed=0):
    """calculates mean bootstrapped ATE for PS and plots bootstrap distribution.
    Use bss of 1000 for when writing up results"""

    np.random.seed(ran_seed)
    ates = Parallel(n_jobs=-1)(delayed(run_ps)(df.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bss))
    ates = np.array(ates)
    print(f"ATE: {ates.mean()}")
    print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")

    sns.distplot(ates, bins=17, kde=False)  # added bins=17 to match earlier plots
    plt.vlines(np.percentile(ates, 2.5), 0, 35, linestyles="dotted")
    plt.vlines(np.percentile(ates, 97.5), 0, 35, linestyles="dotted", label="95% CI")
    plt.title("ATE Bootstrap Distribution: PS")
    plt.legend()
    plt.show()

    controls = 'core plus topic' if 'dominant_topic' in X else 'core'

    return [paper_name, 'propensity score', Y, T, controls, len(df), ates.mean(), np.percentile(ates, 2.5),
            np.percentile(ates, 97.5)] + ['NA']*6

def doubly_robust(df, X, T, Y):
    """computes the doubly robust est within run_dr_model"""
    ps = LogisticRegression(penalty=None, max_iter=10000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

def run_dr_model(df, X, T, Y, paper_name, bss=1000, ran_seed=0):
    """calculates mean bootstrapped ATE for DR and plots bootstrap distribution.
        Use bss of 1000 for when writing up results"""
    np.random.seed(ran_seed)
    # run 1000 bootstrap samples
    ates = Parallel(n_jobs=-1)(delayed(doubly_robust)(df.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bss))
    ates = np.array(ates)
    print(f"ATE: {ates.mean()}")
    print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))

    sns.distplot(ates, bins=17, kde=False)  # added bins=17 to match earlier plots
    plt.vlines(np.percentile(ates, 2.5), 0, 35, linestyles="dotted")
    plt.vlines(np.percentile(ates, 97.5), 0, 35, linestyles="dotted", label="95% CI")
    #plt.title("ATE Bootstrap Distribution: DR")
    plt.legend()
    plt.show()

    controls = 'core plus topic' if 'dominant_topic' in X else 'core'

    return [paper_name, 'doubly robust', Y, T, controls, len(df), ates.mean(), np.percentile(ates, 2.5),
            np.percentile(ates, 97.5)] + ['NA']*6


# Load df.s

root = '/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/'

topics = pd.read_csv(root + 'art_topic.csv')  # topics, all arts all papers
df_tweetsDM = pd.read_csv(root + 'aggDM.csv')  # Daily Mail
df_i_tweetsDM = pd.read_csv(root + 'tweets_aggDM.csv')  # Daily Mail
df_mddDM = pd.read_csv(root + 'mddatcf_dailymail.csv')
df_tweetsG = pd.read_csv(root + 'aggG.csv')  # Guardian
df_i_tweetsG = pd.read_csv(root + 'tweets_aggG.csv')  # Daily Mail
df_mddG = pd.read_csv(root + 'mddatcf_guardian.csv')
df_tweetsNYP = pd.read_csv(root + 'aggNYP.csv')  # New York Post
df_i_tweetsNYP = pd.read_csv(root + 'tweets_aggNYP.csv')  # Daily Mail
df_mddNYP = pd.read_csv(root + 'mddatcf_nypost.csv')
df_i_tweetsNYT = pd.read_csv(root + 'tweets_aggNYT.csv')  # Daily Mail
df_tweetsNYT = pd.read_csv(root + 'aggNYT.csv')  # New York Times
df_mddNYT = pd.read_csv(root + 'mddatcf_nytimes.csv')

dfs_list = [[df_tweetsDM, df_mddDM], [df_tweetsG, df_mddG], [df_tweetsNYP, df_mddNYP], [df_tweetsNYT, df_mddNYT]]

manip_dfs_list = [manip_df(df_mdd=dfs[1], df_tweets=dfs[0], topics=topics) for dfs in dfs_list]  # Create nec var.s and manip data

manip_i_dfs_list = []
for df_i_tweets, manipd_df in zip([df_i_tweetsDM, df_i_tweetsG, df_i_tweetsNYP, df_i_tweetsNYT], manip_dfs_list):
    manipd_i_df = manip_i_df(df_i_tweets, manipd_df)
    manip_i_dfs_list.append(manipd_i_df)

# Apply functions

[print_plot_descs(df) for df in manip_dfs_list]  # Calc core descriptives
# 4635036 + 495708 + 604212 + 257996 tweets (in rev order of paper_names)

X = ['my_word_count_log', 'mean_my_word_len_log', 'mean_my_words_in_sen_log', 'year_2020', 'year_2021', 'day_of_week_1',
     'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'month_02', 'month_03',
     'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12']
T = 'neg_art'
Y = 'n_tweets_log'


paper_names = ['DM', 'G', 'NYP', 'NYT']
out_groups = ['maj_lab', 'maj_con', 'maj_dem', 'maj_rep']
in_groups = ['maj_con', 'maj_lab', 'maj_rep', 'maj_dem']
topic_nums = range(len(topics['dominant_topic'].unique()))

analyses_dict = {}

# main models
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core{paper_name}'] = run_standard_reg(Y='n_tweets_log_CHECK', T=T, X=X, df=manip_dfs_list[i], paper_name=paper_name)

# robustness checks, using alt neg art defin
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_rob{paper_name}'] = run_standard_reg(Y=Y, T='neg_art_r', X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# PS 'robustness' check models
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_PSrob{paper_name}'] = run_prop_score_model(Y=Y, T=T, X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# DR 'robustness' check models
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_DRrob{paper_name}'] = run_dr_model(Y=Y, T=T, X=X, df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# core models with topic controls 'robustness' check models
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_topic_cont{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X + ['dominant_topic'], df=manip_dfs_list[i],
                                                               paper_name=paper_name)

# dividing by topic
for i, paper_name in enumerate(paper_names):
    for j in topic_nums:
        by_topic_df = manip_dfs_list[i][manip_dfs_list[i]['dominant_topic'] == j]
        analyses_dict[f'MR_core_by_topic{j}{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X, df=by_topic_df,
                                                                       paper_name=paper_name)

# main models but with subset df to facil comparison with title models
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_title_subset{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X, df=manip_dfs_list[i].dropna(),
                                                                paper_name=paper_name)

# title model
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_title{paper_name}'] = run_standard_reg(Y=Y, T='neg_title', X=X, df=manip_dfs_list[i].dropna(),
                                                               paper_name=paper_name)

# art and title model
# Note: prob not for inclusion in write-up as likely collinearity between neg art and neg title
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_art_title{paper_name}'] = run_standard_reg(Y=Y, T='neg_art+neg_title', X=X, df=manip_dfs_list[i].dropna(),
                                                               paper_name=paper_name)

# out-group models
for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    analyses_dict[f'MR_out_group{paper_name}'] = run_standard_reg(Y=Y, T=out_groups[i], X=X,
                                                                  df=manip_dfs_list[i][manip_dfs_list[i][any_nation]==1],
                                                                  paper_name=paper_name)

for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    analyses_dict[f'MR_in_group{paper_name}'] = run_standard_reg(Y=Y, T=in_groups[i], X=X,
                                                                  df=manip_dfs_list[i][manip_dfs_list[i][any_nation]==1],
                                                                  paper_name=paper_name)

for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    T = 'neg_art+' + out_groups[i] + '+neg_art*' + out_groups[i]
    analyses_dict[f'MR_neg_out_int{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X,
                                                                  df=manip_dfs_list[i][manip_dfs_list[i][any_nation]==1],
                                                                  paper_name=paper_name)

for i, paper_name in enumerate(paper_names):
    any_nation = 'any_pol_pers_term_UK' if paper_name in ['DM', 'G'] else 'any_pol_pers_term_US'
    T = 'neg_art+' + in_groups[i] + '+neg_art*' + in_groups[i]
    analyses_dict[f'MR_neg_in_int{paper_name}'] = run_standard_reg(Y=Y, T=T, X=X,
                                                                  df=manip_dfs_list[i][manip_dfs_list[i][any_nation]==1],
                                                                  paper_name=paper_name)

# predicting RT of tweets re neg arts
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_LogRTs{paper_name}'] = run_standard_reg(Y='n_RTs_log', T='neg_art', X=X+['char_length_tweet_log'],
                                                                 df=manip_i_dfs_list[i], paper_name=paper_name)

# predicting RT of tweets using title
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_LogRTs{paper_name}'] = run_standard_reg(Y='n_RTs_log', T='neg_title',
                                                                    X=X+['char_length_tweet_log'],
                                                                    df=manip_i_dfs_list[i], paper_name=paper_name)

# core but predicting tweets+RTs
for i, paper_name in enumerate(paper_names):
    analyses_dict[f'MR_core_pred_tweetsRTs{paper_name}'] = run_standard_reg(Y='n_tweetsPlusRTs_log', T='neg_art', X=X, df=manip_dfs_list[i], paper_name=paper_name)

# internal meta-analysis... to be added

all_results = pd.DataFrame(analyses_dict).T
all_results.columns = ['paper', 'analysis_method', 'y_variable', 'key_predictors', 'controls', 'sample_size',
                       'point_est1', 'lower_est1', 'upper_est1', 'point_est2', 'lower_est2', 'upper_est2', 'point_est3',
                       'lower_est3', 'upper_est3']

#all_results.to_csv(root + 'all_results17062023.csv')  # with 'fascist' removed from cons identity words (due to
# presence in Vader dict)
#all_results.to_csv(root + 'all_results07062023.csv')  # with 'fascist' erroneously left in cons identity words, and
# therefore containing some models with multicollinearity problems
