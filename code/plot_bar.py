from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] =  1.5
plt.rcParams['ytick.major.width'] =  1.5
plt.rcParams['figure.figsize'] =  (17, 9)

labels = ['Proposed', 'MEF+GMM', 'MEF+OCSVM', 'MEF+SVDD', 'MEF+LOF', 'MEF+IF', 'CNN+GMM', 'CNN+OCSVM', 'CNN+SVDD', 'CNN+LOF', 'CNN+IF']
# men_means = [91.37, 24.50, 49.69, 59.52, 66.92, 68.71, 63.06, 47.10, 81.26, 92.41, 75.85] # cwru已知，更新前
# women_means = [90.57, 95.83, 98.11, 63.18, 40.00, 68.51, 92.24, 94.65, 86.56, 79.95, 89.31] # cwru未知，更新前
# women_means = [88.06, 24.00, 49.66, 46.08, 43.96, 59.85, 54.91, 44.69, 78.60, 88.45, 71.28] # cwru已知，更新后

men_means = [96.59, 45.31, 40.1, 52.61, 70.14, 65.92, 66.56, 48.84, 86.19, 95.8, 80.44] # RMSF已知，更新前
# women_means = [95.33, 100, 93.25, 75.14, 77.63, 89.64, 94.42, 95.89, 90.05, 84.4, 90.45] # RMSF未知，更新前
women_means = [95.77, 45.45, 37.86, 43.49, 63.75, 62, 59, 46.74, 84.9, 93.49, 76.65] # RMSF已知，更新后
 
average = np.mean(np.array(men_means))*0.5 + np.mean(np.array(women_means))*0.5

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Before model self-learning', color='#FFFF99')
rects2 = ax.bar(x + width/2, women_means, width, label='After model self-learning', color='#99CCFF')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20)
ax.set_ylim(25, 105)
ax.axhline(y=average, ls='--', color='k')
ax.legend(loc='upper center')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=0)


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig('../img/rmsf_KNOWN_self-learning.png')
# plt.show()