import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

fontsize_title = 18
fontsize_label = 14
fontsize_legend = 14

labels = ['w/o trap', 'w/ trap', 'oocyte']
raw = [67.8, 56.9, 54.96]
scat = [79.66, 78.74, 78.88]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, raw, width, label='Raw')
rects2 = ax.bar(x + width/2, scat, width, label='Scat')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)', fontsize=fontsize_label)
ax.set_title('Classification accuracy', fontsize=fontsize_title)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fontsize_label)
ax.set_ylim([0, 110])
ax.legend(loc='upper right', fontsize=fontsize_legend)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)

#fig.tight_layout()

plt.show()
