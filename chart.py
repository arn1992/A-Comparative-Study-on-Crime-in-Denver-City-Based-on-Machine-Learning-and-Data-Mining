import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rc('ytick',labelsize=18)
labels = ['1D-Inception', 'BD-LSTM']
men_means = [60.65, 57.09]
women_means = [68.40, 65.86]
boy_means = [71.18, 69.46]

N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars
x = np.arange(len(labels))  # the label locations
#width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15,10))
rects1 = ax.bar(x, men_means, width, label='Sequence Features',color='black')
rects2 = ax.bar(x+width, women_means, width, label='Profile Features',color='blue')
rects3 = ax.bar(x+width*2, boy_means, width, label='Sequence and Profile Features',color='gray')



def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=20)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
# Add some text for labels, title and custom x-axis tick labels, etc.

#ax.set_title('man')
ax.set_xticks(x+width)
ax.set_xticklabels(labels,fontsize=20)
ax.set_ylabel('Q8 Accuracy (%)',fontsize=20)
ax.legend( fontsize=18,loc='upper center')

fig.tight_layout()


plt.show()

