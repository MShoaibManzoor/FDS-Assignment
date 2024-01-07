import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt("./data5-1.csv")

n_bins = 25

# Generating a plotable distribution of data & storing edge values of each bin.
dist, bin_edges = np.histogram(
    data, bins=n_bins, range=[0.0, 110000.0])

# Calculating the center value & width of distributions each bin.
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = bin_edges[1:] - bin_edges[:-1]

# Normalizing the distribution to obtain PDF.
pdf = dist / np.sum(dist)

# Calculating mean of distribution by summing
# center values multiplied by their density.
mean = np.sum(bin_centers*pdf)

# Calculating the cumulative distribution for X = 10th_percentile.
cumulative_dist = np.cumsum(pdf)

# Calculating values below 10th percentile.
index_below_10th = np.argmin(np.abs(cumulative_dist-0.10))
X = bin_edges[index_below_10th]

plt.figure(figsize=(12, 6))

plt.bar(bin_centers, pdf, width=0.85*bin_widths,
        color='teal', alpha=0.68, label='PDF')
plt.bar(bin_centers[0:index_below_10th], pdf[0:index_below_10th],
        width=0.85*bin_widths, color='maroon', alpha=0.8, label='10th percentile')
plt.axvline(mean, color='red', alpha=0.8,
            linestyle='dashed', label='Mean value')
plt.axvline(X, color='maroon',
            linestyle='dotted')
text = f'<=Mean: {mean: .2f}'
text_percentile = f'<=X(10th_percentile): {X: .2f}'
plt.text(mean+1000, y=np.mean(pdf)+0.04, s=text, fontsize=11.5)
plt.text(X+1000, y=np.mean(pdf), s=text_percentile, fontsize=11.5)

plt.xlabel('Salaries, €', fontsize=14)
plt.ylabel('Probable Density', fontsize=14)
plt.suptitle('Density of Annual Salaries',
             fontsize=18, fontweight='bold')
plt.legend(frameon=False, fontsize=11, ncols=4,
           loc='upper center', borderaxespad=-2.4)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10.5)

plt.savefig('22038495.png')
