import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

def plot_logged_data(file_path='/home/amandoee/sim_ws/logged_data.csv'):
    timestamps = []
    current_x = []
    current_y = []
    ground_truth_x = []
    ground_truth_y = []
    headings = []
    laps = []

    # Read the logged data from the CSV file
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamps.append(row['timestamp'])
            current_x.append(float(row['current_x']))
            current_y.append(float(row['current_y']))
            ground_truth_x.append(float(row['ground_truth_x']))
            ground_truth_y.append(float(row['ground_truth_y']))
            headings.append(float(row['ground_truth_yaw']))  # Assuming 'heading' column exists in the CSV
            laps.append(int(row['lap']))  # Assuming 'lap' column exists in the CSV

    # Calculate absolute mean error (AME) and per-point errors
    errors = [
        np.sqrt((gx - cx)**2 + (gy - cy)**2)
        for gx, gy, cx, cy in zip(ground_truth_x, ground_truth_y, current_x, current_y)
    ]
    absolute_mean_error = np.mean(errors)

    # Calculate local frame errors (forward and lateral)
    forward_errors = []
    lateral_errors = []
    for i in range(len(ground_truth_x) - 1):
        # Calculate heading direction from ground truth
        heading=  headings[i]
        
        # Calculate error vector
        ex = current_x[i] - ground_truth_x[i]
        ey = current_y[i] - ground_truth_y[i]

        # Transform error into local frame
        forward_error = ex * np.cos(heading) + ey * np.sin(heading)
        lateral_error = -ex * np.sin(heading) + ey * np.cos(heading)

        forward_errors.append(forward_error)
        lateral_errors.append(lateral_error)

    # Mean forward and lateral errors
    mean_forward_error = np.mean(np.abs(forward_errors))
    mean_lateral_error = np.mean(np.abs(lateral_errors))

    # Create a colormap for laps
    cmap = get_cmap('viridis', np.max(laps) + 1)

    # Plot the data
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Plot ground truth vs estimated positions with lap-based colors
    for lap in range(np.min(laps), np.max(laps) + 1):
        lap_indices = [i for i, l in enumerate(laps) if l == lap]
        axs[0, 0].plot(
            [ground_truth_x[i] for i in lap_indices],
            [ground_truth_y[i] for i in lap_indices],
            label=f'Ground Truth Lap {lap}',
            color=cmap(lap),
            marker='o'
        )
        axs[0, 0].plot(
            [current_x[i] for i in lap_indices],
            [current_y[i] for i in lap_indices],
            label=f'Estimated Lap {lap}',
            color=cmap(lap),
            linestyle='--',
            marker='x'
        )

    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylabel('Y Position')
    axs[0, 0].set_title(f'Ground Truth vs Estimated Path\nAbsolute Mean Error: {absolute_mean_error:.2f}')
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Plot forward errors
    axs[0, 1].plot(forward_errors, label=f'Mean Forward Error: {mean_forward_error:.2f}', color='blue')
    axs[0, 1].set_xlabel('Point Index')
    axs[0, 1].set_ylabel('Forward Error')
    axs[0, 1].set_title('Forward Errors (Along Heading)')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot lateral errors
    axs[1, 0].plot(lateral_errors, label=f'Mean Lateral Error: {mean_lateral_error:.2f}', color='green')
    axs[1, 0].set_xlabel('Point Index')
    axs[1, 0].set_ylabel('Lateral Error')
    axs[1, 0].set_title('Lateral Errors (Perpendicular to Heading)')
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Call the function to plot the data
plot_logged_data()