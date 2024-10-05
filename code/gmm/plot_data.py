import data
import matplotlib.pyplot as plt

datasets = ['halfmoons', 'spirals', 'board']

for d in datasets:
    x = data.get_2d(d, 10000 if d == 'board' else 6000, 42)
    # plt.figure(figsize=(8, 5))
    plt.scatter(x[:, 0], x[:, 1], 
                c='dodgerblue',    # Set point color
                s=20,              # Set point size
                alpha=0.6,         # Add transparency to points
                edgecolor='k',      # Black edge for contrast
                linewidth=0.1)      # Thin edge

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{d}.png", format='png', bbox_inches='tight')
    plt.clf()