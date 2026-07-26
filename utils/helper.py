import numpy as np
import matplotlib.pyplot as plt

def plot_mlp_boundary(X, Y, model, mesh_step=0.02):
    # 1. Establish canvas limits with a 0.5 margin padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 2. Generate the dense 2D coordinate meshgrid arrays
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))
    
    # 3. Flatten grid points to prepare for batch processing
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 4. Use the MLP predict method to cleanly fetch class assignments (0 or 1)
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 5. Render the visual elements
    plt.figure(figsize=(8, 6))
    
    # Fill background with 'Spectral' color splits (red/blue halves)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=1.0)
    
    # Scatter true data points on top with standard outlines
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral, edgecolors='black', s=25)
    
    # Clean up graph borders and strip standard tick markings
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    plt.show()

# --- How to call it after training ---
# plot_mlp_boundary(X, Y, model)
