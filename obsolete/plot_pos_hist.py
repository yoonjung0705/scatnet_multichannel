'''module that plots the histogram of the measured qpd position'''

plt.close('all') 
fig, ax = plt.subplots() 
fig2, ax2 = plt.subplots() 
 
led_levels = ['0p5', '1p0', '1p5', '2p0', '2p5'] 
for led_level in led_levels: 
    file_name = 'ad57_8x_led_{}_bead_3um_laser_100ma_1.txt'.format(led_level) 
    df = pd.read_csv(file_name, header=None, delimiter="\t") 
    x = df.loc[:, 0].values 
    y = df.loc[:, 1].values 
    x_var = x[::stepsize].var() 
    y_var = y[::stepsize].var() 
    ax.hist(x, label=led_level, bins=30) 
    ax2.hist(y, label=led_level, bins=30) 

ax.legend() 
ax2.legend() 

