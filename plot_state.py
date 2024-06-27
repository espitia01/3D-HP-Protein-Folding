import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import OrderedDict

def plot_conformation(info, save_path=""):
    sns.set(style="ticks", font_scale=1.5, context="paper", rc={"lines.linewidth": 2.5})
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    state_chain = list(info['state_chain'].items())
    x, y, z = zip(*[coord for coord, _ in state_chain])
    types = [t for _, t in state_chain]
    
    # Plot backbone with equal length segments
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color='#1f77b4', linewidth=3, alpha=0.8)
    
    # Plot amino acids
    H_coords = [(xi, yi, zi) for (xi, yi, zi), t in state_chain if t == 'H']
    P_coords = [(xi, yi, zi) for (xi, yi, zi), t in state_chain if t == 'P']
    
    H_x, H_y, H_z = zip(*H_coords)
    P_x, P_y, P_z = zip(*P_coords)
    
    ax.scatter(H_x, H_y, H_z, color='#ff7f0e', s=300, label="Hydrophobic (H)", depthshade=False, edgecolors='black', linewidths=1.5)
    ax.scatter(P_x, P_y, P_z, color='#2ca02c', s=300, label="Polar (P)", depthshade=False, edgecolors='black', linewidths=1.5)
    
    # Plot H-H bonds (only adjacent H-H bonds not connected by the backbone)
    plotted_hh_bond = False
    for i, ((xi, yi, zi), ti) in enumerate(state_chain):
        if ti == 'H':
            for j, ((xj, yj, zj), tj) in enumerate(state_chain):
                if tj == 'H' and i != j and abs(i - j) > 1:
                    dist = ((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**0.5
                    if dist == 1:
                        if not plotted_hh_bond:
                            ax.plot([xi, xj], [yi, yj], [zi, zj], color='#d62728', linestyle='--', linewidth=1, label="H-H Bond", alpha=0.7)
                            plotted_hh_bond = True
                        else:
                            ax.plot([xi, xj], [yi, yj], [zi, zj], color='#d62728', linestyle='--', linewidth=1, alpha=0.7)
    
    #ax.set_xlabel('X', fontsize=20, labelpad=15)
    #ax.set_ylabel('Y', fontsize=20, labelpad=15)
    #ax.set_zlabel('Z', fontsize=20, labelpad=15)
   #plt.title("3D Protein Conformation", fontsize=24, pad=20, fontweight='bold')
    #ax.legend(loc='upper right', fontsize=16, frameon=True, facecolor='white', edgecolor='black')
        
    # Adjust grid and tick settings
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Adjust grid and tick settings
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    #ax.tick_params(axis='both', which='major', labelsize=16, pad=10)
        
    # Adjust 3D view angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}conformation.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

info = {
    'state_chain': OrderedDict([((0, 0, 0), 'H'), ((0, 0, -1), 'P'), ((-1, 0, -1), 'H'), ((-2, 0, -1), 'P'), ((-2, 0, 0), 'P'), ((-1, 0, 0), 'H'), ((-1, 1, 0), 'H'), ((-1, 1, 1), 'P'), ((-1, 1, 2), 'H'), ((-2, 1, 2), 'P'), ((-2, 1, 1), 'P'), ((-2, 1, 0), 'H'), ((-2, 1, -1), 'P'), ((-1, 1, -1), 'H'), ((0, 1, -1), 'H'), ((1, 1, -1), 'P'), ((1, 1, 0), 'P'), ((0, 1, 0), 'H'), ((0, 2, 0), 'P'), ((-1, 2, 0), 'H')])
}
plot_conformation(info, save_path="./")