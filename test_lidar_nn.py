import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import from lidar_NN.py
try:
    from lidar_NN import LocalizationNet
except ImportError:
    # Define a minimal version of LocalizationNet if import fails
    class LocalizationNet(nn.Module):
        def __init__(self):
            super(LocalizationNet, self).__init__()
            # LiDAR branch: Input is a 1D signal of length 1080
            self.conv_layers = nn.Sequential(
                # Input: (batch, 1, 1080)
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),  # output: (batch, 16, 540)
                
                nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),  # output: (batch, 32, 270)
                
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)   # output: (batch, 64, 135)
            )
            # Fully connected part: flatten and regress to 3 outputs (x, y, yaw)
            self.fc = nn.Sequential(
                nn.Linear(1080, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
            
        def forward(self, lidar):
            # lidar: (batch, 1080) -> add channel dimension: (batch, 1, 1080)
            x = lidar.unsqueeze(1)
            #x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # flatten
            out = self.fc(x)
            return out

# Create dataset class
class LidarDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        print(f"Loading data from {csv_path}")
        # Read the CSV file
        self.df = pd.read_csv(csv_path, sep=';')
        
        # Create lists to hold the LiDAR scans and targets
        self.lidar_data = []
        self.targets = []
        
        # Process data from consecutive rows
        for idx in range(len(self.df) - 1):
            row1 = self.df.iloc[idx]
            row2 = self.df.iloc[idx+1]
            scan_str = row1['lidar_scan']
            scan_str2 = row2['lidar_scan']
            pos_x = row1['x'] - row2['x']
            pos_y = row1['y'] - row2['y']

            yaw = row1['yaw']- row2['yaw']

            # Extract the list of ranges from the string, e.g. "ranges=[...]"
            ranges = np.array(literal_eval(f'{scan_str}'))
            ranges2 = np.array(literal_eval(f'{scan_str2}'))
            d_ranges = ranges - ranges2
            # print("test", pos_x,pos_y)
            full_range = [0 for i in range(1350)]
            for index, i in enumerate(range(round(row1['yaw']/0.0043633),round(1080+row1['yaw']/0.0043633))):
                if index >= 1080:
                    break
                full_range[i % 1350] = d_ranges[index]

            d_ranges = np.append(full_range, row1['yaw'])
            self.lidar_data.append(d_ranges)
            self.targets.append([pos_x,pos_y,yaw])
        
        # Convert to numpy arrays
        self.lidar_data = np.array(self.lidar_data)
        self.targets = np.array(self.targets)
        
        print(f"Dataset loaded with {len(self.lidar_data)} samples")
        print(f"LiDAR data shape: {self.lidar_data.shape}")
        print(f"Target data shape: {self.targets.shape}")
        
    def __len__(self):
        return len(self.lidar_data)
    
    def __getitem__(self, idx):
        lidar_diff = torch.tensor(self.lidar_data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return lidar_diff, target

def load_trained_model(model_path='localization_model.pth'):
    """Load the trained model from the checkpoint file."""
    try:
        model = LocalizationNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def reconstruct_trajectory(predictions, actual, initial_pose=(0, 0, 0)):
    """
    Reconstruct the full trajectory from incremental pose changes.
    
    Args:
        predictions: Numpy array of predicted changes in x, y, yaw
        actual: Numpy array of actual changes in x, y, yaw
        initial_pose: Initial pose (x, y, yaw) to start from
    
    Returns:
        pred_trajectory: List of (x, y, yaw) for predicted trajectory
        true_trajectory: List of (x, y, yaw) for actual trajectory
    """
    pred_trajectory = [list(initial_pose)]
    true_trajectory = [list(initial_pose)]
    
    for i in range(len(predictions)):
        # Calculate next predicted pose
        pred_x = pred_trajectory[-1][0] + predictions[i][0]
        pred_y = pred_trajectory[-1][1] + predictions[i][1]
        pred_yaw = pred_trajectory[-1][2] + predictions[i][2]
        pred_trajectory.append([pred_x, pred_y, pred_yaw])
        
        # Calculate next actual pose
        true_x = true_trajectory[-1][0] + actual[i][0]
        true_y = true_trajectory[-1][1] + actual[i][1]
        true_yaw = true_trajectory[-1][2] + actual[i][2]
        true_trajectory.append([true_x, true_y, true_yaw])
    
    return np.array(pred_trajectory), np.array(true_trajectory)

def plot_trajectory_comparison(pred_trajectory, true_trajectory, save_path='trajectory_comparison.png'):
    """
    Plot the predicted and actual trajectories for comparison.
    """
    plt.figure(figsize=(12, 10))
    
    # Plot trajectories
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'b-', label='Predicted Trajectory', linewidth=2)
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', label='Ground Truth Trajectory', linewidth=2)
    
    # Mark start and end points
    plt.scatter(pred_trajectory[0, 0], pred_trajectory[0, 1], c='blue', s=100, marker='^', label='Start Point')
    plt.scatter(pred_trajectory[-1, 0], pred_trajectory[-1, 1], c='red', s=100, marker='o', label='End Point (Predicted)')
    plt.scatter(true_trajectory[-1, 0], true_trajectory[-1, 1], c='green', s=100, marker='o', label='End Point (Ground Truth)')
    
    # Add direction arrows periodically
    arrow_indices = np.linspace(0, len(pred_trajectory)-1, 20, dtype=int)
    for i in arrow_indices:
        # Predicted trajectory direction arrow
        dx = np.cos(pred_trajectory[i, 2]) * 0.2
        dy = np.sin(pred_trajectory[i, 2]) * 0.2
        plt.arrow(pred_trajectory[i, 0], pred_trajectory[i, 1], dx, dy, 
                  head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
        
        # Ground truth trajectory direction arrow
        dx = np.cos(true_trajectory[i, 2]) * 0.2
        dy = np.sin(true_trajectory[i, 2]) * 0.2
        plt.arrow(true_trajectory[i, 0], true_trajectory[i, 1], dx, dy, 
                  head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.7)
    
    plt.title('Predicted vs. Ground Truth Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory comparison saved to {save_path}")

def plot_error_distribution(predictions, actual, save_path='error_distribution.png'):
    """
    Plot the error distribution for each prediction component.
    """
    # Calculate errors
    errors = predictions - actual
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot error distributions
    for i, label in enumerate(['X Error', 'Y Error', 'Yaw Error']):
        sns.histplot(errors[:, i], kde=True, ax=axs[i])
        axs[i].set_title(label)
        axs[i].grid(True)
        
        # Calculate statistics
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])
        rmse = np.sqrt(np.mean(errors[:, i] ** 2))
        
        # Add statistics to plot
        stats_text = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nRMSE: {rmse:.4f}'
        axs[i].text(0.05, 0.95, stats_text, transform=axs[i].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error distribution plots saved to {save_path}")

def plot_cumulative_error(pred_trajectory, true_trajectory, save_path='cumulative_error.png'):
    """
    Plot the cumulative error over time to visualize drift.
    """
    # Calculate position error at each step
    position_errors = np.sqrt(
        (pred_trajectory[:, 0] - true_trajectory[:, 0])**2 + 
        (pred_trajectory[:, 1] - true_trajectory[:, 1])**2
    )
    
    # Calculate orientation error at each step (handle circular difference)
    orientation_errors = np.abs(pred_trajectory[:, 2] - true_trajectory[:, 2])
    orientation_errors = np.minimum(orientation_errors, 2*np.pi - orientation_errors)
    orientation_errors = np.rad2deg(orientation_errors)  # Convert to degrees
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot position error
    axs[0].plot(position_errors, 'b-', linewidth=2)
    axs[0].set_ylabel('Position Error (units)')
    axs[0].set_title('Cumulative Position Error')
    axs[0].grid(True)
    
    # Plot orientation error
    axs[1].plot(orientation_errors, 'r-', linewidth=2)
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Orientation Error (degrees)')
    axs[1].set_title('Cumulative Orientation Error')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cumulative error plots saved to {save_path}")

def plot_prediction_vs_actual(predictions, actual, save_path='prediction_vs_actual.png'):
    """
    Create scatter plots of predicted vs. actual values for each component.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, label in enumerate(['X', 'Y', 'Yaw']):
        axs[i].scatter(actual[:, i], predictions[:, i], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(actual[:, i]), min(predictions[:, i]))
        max_val = max(max(actual[:, i]), max(predictions[:, i]))
        axs[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axs[i].set_xlabel(f'Actual {label}')
        axs[i].set_ylabel(f'Predicted {label}')
        axs[i].set_title(f'{label} - Predicted vs. Actual')
        axs[i].grid(True)
        
        # Calculate and display metrics
        r2 = np.corrcoef(actual[:, i], predictions[:, i])[0, 1]**2
        mse = mean_squared_error(actual[:, i], predictions[:, i])
        mae = mean_absolute_error(actual[:, i], predictions[:, i])
        
        stats_text = f'R²: {r2:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f}'
        axs[i].text(0.05, 0.95, stats_text, transform=axs[i].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction vs. actual plots saved to {save_path}")

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_error_over_distance(pred_trajectory, true_trajectory, save_path='error_over_distance.png'):
    """
    Plot error magnitude as it relates to distance traveled.
    """
    # Calculate cumulative distance traveled
    true_displacements = np.diff(true_trajectory[:, :2], axis=0)
    cumulative_distance = np.zeros(len(true_trajectory))
    cumulative_distance[1:] = np.cumsum(np.sqrt(true_displacements[:, 0]**2 + true_displacements[:, 1]**2))
    
    # Calculate position errors
    position_errors = np.sqrt(
        (pred_trajectory[:, 0] - true_trajectory[:, 0])**2 + 
        (pred_trajectory[:, 1] - true_trajectory[:, 1])**2
    )
    
    # Calculate relative error as a percentage of distance traveled
    relative_errors = np.zeros_like(position_errors)
    nonzero_indices = cumulative_distance > 0
    relative_errors[nonzero_indices] = 100 * position_errors[nonzero_indices] / cumulative_distance[nonzero_indices]
    
    plt.figure(figsize=(12, 6))
    
    # Plot absolute error
    plt.subplot(1, 2, 1)
    plt.scatter(cumulative_distance, position_errors, alpha=0.5)
    plt.plot(cumulative_distance, position_errors, alpha=0.3)
    plt.xlabel('Cumulative Distance Traveled')
    plt.ylabel('Position Error')
    plt.title('Absolute Error vs. Distance Traveled')
    plt.grid(True)
    
    # Add trend line
    z = np.polyfit(cumulative_distance, position_errors, 1)
    p = np.poly1d(z)
    plt.plot(cumulative_distance, p(cumulative_distance), "r--", alpha=0.8)
    plt.text(0.05, 0.95, f'Trend: y = {z[0]:.4f}x + {z[1]:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot relative error
    plt.subplot(1, 2, 2)
    valid_indices = (cumulative_distance > 0.1) & (relative_errors < 100)  # Filter out very small distances and large outliers
    plt.scatter(cumulative_distance[valid_indices], relative_errors[valid_indices], alpha=0.5)
    plt.xlabel('Cumulative Distance Traveled')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error vs. Distance Traveled')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error over distance plots saved to {save_path}")

def visualize_lidar_prediction_sample(dataset, model, indices, save_dir='lidar_prediction_samples'):
    """
    Visualize a few LIDAR scan differences and their predicted vs. actual motion.
    
    Args:
        dataset: The LIDAR dataset
        model: The trained model
        indices: List of indices to visualize
        save_dir: Directory to save the visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for idx in indices:
        lidar_diff, true_motion = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            input_tensor = lidar_diff.unsqueeze(0).to(device)
            pred_motion = model(input_tensor).cpu().numpy()[0]
        
        true_motion = true_motion.numpy()
        
        # Create a figure with 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot LIDAR difference
        angles = np.linspace(0, 2*np.pi, len(lidar_diff), endpoint=False)
        lidar_diff_np = lidar_diff.numpy()
        
        axs[0].plot(angles, lidar_diff_np)
        axs[0].set_title(f'LIDAR Scan Difference (Sample {idx})')
        axs[0].set_xlabel('Angle (radians)')
        axs[0].set_ylabel('Distance Difference')
        axs[0].set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        axs[0].set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
        axs[0].grid(True)
        
        # Plot motion prediction
        axs[1].set_xlim(-0.3, 0.3)
        axs[1].set_ylim(-0.3, 0.3)
        
        # Draw coordinate frame
        axs[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axs[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Draw true motion arrow
        axs[1].arrow(0, 0, true_motion[0], true_motion[1], 
                    head_width=0.02, head_length=0.03, fc='green', ec='green', 
                    label=f'True Motion (dx={true_motion[0]:.4f}, dy={true_motion[1]:.4f}, dθ={true_motion[2]:.4f})')
        
        # Draw predicted motion arrow
        axs[1].arrow(0, 0, pred_motion[0], pred_motion[1], 
                    head_width=0.02, head_length=0.03, fc='blue', ec='blue', 
                    label=f'Predicted Motion (dx={pred_motion[0]:.4f}, dy={pred_motion[1]:.4f}, dθ={pred_motion[2]:.4f})')
        
        # Draw rotation indicator
        rotation_radius = 0.15
        axs[1].add_patch(plt.Circle((0, 0), rotation_radius, fill=False, alpha=0.3))
        
        # True rotation indicator
        true_angle = np.pi/2  # Start at 90 degrees
        true_end_angle = true_angle + true_motion[2]
        
        # Starting point
        start_x = rotation_radius * np.cos(true_angle)
        start_y = rotation_radius * np.sin(true_angle)
        # Displacement
        dx = rotation_radius * np.cos(true_end_angle) - start_x
        dy = rotation_radius * np.sin(true_end_angle) - start_y
        
        axs[1].arrow(start_x, start_y, dx, dy,
                    head_width=0.015, head_length=0.02, fc='green', ec='green', alpha=0.7)
        
        # Predicted rotation indicator
        pred_end_angle = true_angle + pred_motion[2]
        
        # Starting point
        start_x = rotation_radius * np.cos(true_angle)
        start_y = rotation_radius * np.sin(true_angle)
        # Displacement
        dx = rotation_radius * np.cos(pred_end_angle) - start_x
        dy = rotation_radius * np.sin(pred_end_angle) - start_y
        
        axs[1].arrow(start_x, start_y, dx, dy,
                    head_width=0.015, head_length=0.02, fc='blue', ec='blue', alpha=0.7)
        
        axs[1].set_title('Predicted vs. True Motion')
        axs[1].set_xlabel('X Change')
        axs[1].set_ylabel('Y Change')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'lidar_prediction_sample_{idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(indices)} LIDAR prediction samples to {save_dir}/")

def main():
    # Create output directory
    output_dir = 'lidar_NN_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # First try to load dataset
        print("Loading dataset...")
        try:
            dataset = LidarDataset('movement_data/movement_data0.csv')
        except FileNotFoundError:
            print("File 'lidar_data.csv' not found. Checking alternative paths...")
            try:
                # Try alternative paths
                if os.path.exists('./data/lidar_data3.csv'):
                    dataset = LidarDataset('./data/lidar_data3.csv')
                else:
                    print("Could not find lidar_data.csv. Please place it in the current directory or in a 'data' subfolder.")
                    return
            except Exception as e:
                print(f"Could not load dataset: {e}")
                return
        
        # Load the trained model
        model = load_trained_model()
        if model is None:
            print("Could not load model. Please ensure 'localization_model.pth' exists.")
            return
        
        # Create DataLoader for predictions
        batch_size = 1
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Make predictions on the entire dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model.to(device)
        model.eval()
        
        all_predictions = []
        all_actual = []
        
        print("Making predictions...")
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs[:,:1350],inputs[:,1350])
                
                all_predictions.append(outputs.cpu().numpy())
                all_actual.append(targets.cpu().numpy())
        
        # Concatenate batch predictions
        predictions = np.vstack(all_predictions)
        actual = np.vstack(all_actual)
        
        print(f"Generated {len(predictions)} predictions")
        
        # Calculate overall error metrics
        mse = np.mean((predictions - actual) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual), axis=0)
        
        print("\nError Metrics:")
        print(f"RMSE - X: {rmse[0]:.6f}, Y: {rmse[1]:.6f}, Yaw: {rmse[2]:.6f}")
        print(f"MAE - X: {mae[0]:.6f}, Y: {mae[1]:.6f}, Yaw: {mae[2]:.6f}")
        
        # Save metrics to a text file
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write("Localization Neural Network Model Evaluation\n")
            f.write("===========================================\n\n")
            f.write(f"Total samples evaluated: {len(predictions)}\n\n")
            f.write("Error Metrics:\n")
            f.write(f"RMSE - X: {rmse[0]:.6f}, Y: {rmse[1]:.6f}, Yaw: {rmse[2]:.6f}\n")
            f.write(f"MAE - X: {mae[0]:.6f}, Y: {mae[1]:.6f}, Yaw: {mae[2]:.6f}\n")
        
        # Generate various plots
        print("\nGenerating plots...")
        
        # 1. Trajectory Comparison
        pred_trajectory, true_trajectory = reconstruct_trajectory(predictions, actual)
        plot_trajectory_comparison(
            pred_trajectory, true_trajectory, 
            save_path=os.path.join(output_dir, 'trajectory_comparison.png')
        )
        
        # 2. Error Distribution
        plot_error_distribution(
            predictions, actual, 
            save_path=os.path.join(output_dir, 'error_distribution.png')
        )
        
        # 3. Cumulative Error
        plot_cumulative_error(
            pred_trajectory, true_trajectory, 
            save_path=os.path.join(output_dir, 'cumulative_error.png')
        )
        
        # 4. Prediction vs. Actual
        plot_prediction_vs_actual(
            predictions, actual, 
            save_path=os.path.join(output_dir, 'prediction_vs_actual.png')
        )
        
        # 5. Error Over Distance
        plot_error_over_distance(
            pred_trajectory, true_trajectory, 
            save_path=os.path.join(output_dir, 'error_over_distance.png')
        )
        
        # 6. Visualize individual LIDAR scan differences and predictions
        # Choose 5 samples evenly distributed through the dataset
        sample_indices = np.linspace(0, len(dataset)-1, 5, dtype=int)
        visualize_lidar_prediction_sample(
            dataset, model, sample_indices, 
            save_dir=os.path.join(output_dir, 'lidar_samples')
        )
        
        print(f"\nAll visualizations saved to the {output_dir}/ directory")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
