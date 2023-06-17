import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import numpy as np


root = '/Users/jw/Desktop/dt/jbs_work/Psychometrician_position/fun_proj_attempts/neg_eng23_files/'
results = pd.read_csv(root + 'all_results17062023.csv', index_col=0)

results['error_diff1'] = results['point_est1'] - results['lower_est1']
results['error_diff2'] = results['point_est2'] - results['lower_est2']
results['error_diff3'] = results['point_est3'] - results['lower_est3']


# PLOT 1

barWidth = 1
bars = 100*results.iloc[0:4, 6]
yer = 100*results.iloc[0:4, -3]
r = np.arange(len(bars))
papers = ['Daily Mail', 'Guardian', 'New York Post', 'New York Times']

patterns = ['x', '.', '', '///']  # Patterns for different papers
colors = ['white', 'white', 'white', 'white']  # Colors for different papers
# colors put as all white, can be reverted to white/grey alternated

fig, ax = plt.subplots(figsize=(8, 5))

for i, (bar, error, pattern, color, paper) in enumerate(zip(bars, yer, patterns, colors, papers)):
    ax.barh(i, bar, height=barWidth, color=color, edgecolor='black', xerr=error, capsize=7, hatch=pattern, label=paper, alpha=0.7)

plt.xlabel('Increase in original tweets for negative articles (%)')
# handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(loc='upper right')  # incorrect order, but running the above line then inc.ing:
# handles[::-1], labels[::-1]  # in ax.legend brackets causes problems with other plots
ax.yaxis.set_tick_params(labelbottom=False)
ax.set_yticks([])
plt.tight_layout()
plt.show()


# PLOT 2

barWidth = 1
papers = ['Daily\nMail', 'Guardian', 'New York\nPost', 'New York\nTimes']
# colors = ['gray', 'white', 'gray', 'white']  # Colors for different papers

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#fig, axs = plt.subplot_mosaic([['A', 'B'], ['C', 'C']], layout='constrained')

for i, ax in enumerate(axs.flat):
    bars = 100 * results.iloc[i*4:(i*4)+4, 6]
    yer = 100 * results.iloc[i*4:(i*4)+4, -3]
    r = np.arange(len(bars))

    for j, (bar, error, pattern, color, paper) in enumerate(zip(bars, yer, patterns, colors, papers)):
        ax.barh(j, bar, height=barWidth, color=color, edgecolor='black', label=paper, xerr=error, capsize=7, hatch=pattern, alpha=0.7)

    ax.set_xlim([0, 39.9])
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])

    if i < 2:
        ax.xaxis.set_tick_params(labelleft=False)
        if i == 0:
            ax.legend(labelspacing=0.8)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='upper right')
    else:
        if i == 2:
            ax.set_xlabel('Increase in original tweets for negative articles (%)')

    label = chr(65 + i)
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom')

plt.tight_layout()
plt.show()


# PLOT 3

label_list = ['A. Sport', 'B. Politics', 'C. Family and Home', 'D. Local', 'E. Global']

fig, axs = plt.subplots(3, 2, figsize=(10, 8))

for i, ax in enumerate(axs.flat):

    if i < 5:
        bars = 100 * results.iloc[20+i*5:25+i*5, 6]
        yer = 100 * results.iloc[20+i*5:25+i*5, -3]
        r = np.arange(len(bars))

        for j, (bar, error, pattern, color, paper) in enumerate(zip(bars, yer, patterns, colors, papers)):
            ax.barh(j, bar, height=barWidth, color=color, edgecolor='black', label=paper, xerr=error, capsize=7, hatch=pattern, alpha=0.7)

        ax.set_xlim([-30, 58])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_yticks([])

        if i < 3:
            ax.xaxis.set_tick_params(labelleft=False)
            if i == 0:
                ax.legend(labelspacing=0.8)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(loc='upper right')
        else:
            if i == 4:
                ax.set_xlabel('Increase in original tweets for negative articles (%)')

        label = label_list[i]
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.5, 1.0, label, transform=ax.transAxes + trans,
                fontsize='medium', va='bottom')

axs[2, 1].remove()  # leave bottom right blank
plt.tight_layout()
plt.show()


# PLOT 4

barWidth = 0.4
bars1 = 100*results.iloc[40:44, 6]
yer1 = 100*results.iloc[40:44, -3]
bars2 = 100*results.iloc[44:48, 6]
yer2 = 100*results.iloc[44:48, -3]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.barh(r1, bars1, height=barWidth, color='white', edgecolor='black', xerr=yer1, capsize=7, label='Article Text')
plt.barh(r2, bars2, height=barWidth, color='grey', edgecolor='black', xerr=yer2, capsize=7, label='Title Text')

plt.yticks([r + barWidth/2 for r in range(len(bars1))], papers)
plt.xlabel('Increase in original tweets for negative articles or titles (%)')
plt.legend()

plt.show()


# PLOT 5  # may not be helpful to include

barWidth = 1
bars = 100*results.iloc[68:72, 6]
yer = 100*results.iloc[68:72, -3]
r = np.arange(len(bars))
papers = ['Daily Mail', 'Guardian', 'New York Post', 'New York Times']

patterns = ['x', '.', '', '///']  # Patterns for different papers
colors = ['white', 'white', 'white', 'white']  # Colors for different papers
# colors put as all white, can be reverted to white/grey alternated

fig, ax = plt.subplots(figsize=(8, 5))

for i, (bar, error, pattern, color, paper) in enumerate(zip(bars, yer, patterns, colors, papers)):
    ax.barh(i, bar, height=barWidth, color=color, edgecolor='black', xerr=error, capsize=7, hatch=pattern, label=paper, alpha=0.7)

plt.xlabel('Increase in retweets for tweets about negative articles (%)')
# handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(loc='upper right')  # incorrect order, but running the above line then inc.ing:
# handles[::-1], labels[::-1]  # in ax.legend brackets causes problems with other plots
ax.yaxis.set_tick_params(labelbottom=False)
ax.set_yticks([])
plt.tight_layout()
plt.show()


# PLOT 6

barWidth = 1
bars = 100*results.iloc[72:76, 6]
yer = 100*results.iloc[72:76, -3]
r = np.arange(len(bars))
papers = ['Daily Mail', 'Guardian', 'New York Post', 'New York Times']

patterns = ['x', '.', '', '///']  # Patterns for different papers

fig, ax = plt.subplots(figsize=(8, 5))

for i, (bar, error, pattern, color, paper) in enumerate(zip(bars, yer, patterns, colors, papers)):
    ax.barh(i, bar, height=barWidth, color=color, edgecolor='black', xerr=error, capsize=7, hatch=pattern, label=paper, alpha=0.7)

plt.xlabel('Increase in original tweets for negative articles (%)')
# handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(loc='upper right')  # incorrect order, but running the above line then inc.ing:
# handles[::-1], labels[::-1]  # in ax.legend brackets causes problems with other plots
ax.yaxis.set_tick_params(labelbottom=False)
ax.set_yticks([])
ax.set_xlim([0, 67])
plt.tight_layout()
plt.show()

# # # Explanation for considering Plot 5 against Plot 6
# If you're predicting Y and find some effect, then just double Y, the coefficient for T will be doubled.
# So, even in cases where tweets get RTd a little less when they concern negative articles:
# There are a higher proportion of tweets concerning neg articles,
# so, effectively multiplying our Y variable by including RTs - even if using a slightly higher
# multiplier for tweets re pos articles - would still lead to an increase in our T coefficient.

