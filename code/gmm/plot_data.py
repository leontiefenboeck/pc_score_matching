import data
import matplotlib.pyplot as plt

datasets = ['halfmoons', 'spirals', 'board']
fig, ax = plt.subplots(1, 3, figsize=(30, 6))

for d, a in zip(datasets, ax):
    x = data.get_2d(d, 10000, 42)
    a.scatter(x[:, 0], x[:, 1], 
                c='dodgerblue',    # Set point color
                s=20,              # Set point size
                alpha=0.7,         # Add transparency to points
                edgecolor='k',      # Black edge for contrast
                linewidth=0.1)      # Thin edge

    # Remove ticks and labels as per original
    a.set_title(f'"{d}"', fontsize=30, fontweight='bold', pad=20)
    a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # a.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    a.axis('off')

plt.savefig(f"datasets.png", format='png')