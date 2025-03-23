import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from matplotlib.colors import LinearSegmentedColormap

@dataclass
class SoccerPitchConfiguration:
    width: int = 7000  # [cm]
    length: int = 12000  # [cm]
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

def draw_pitch_with_heatmap(positions):
    pitch = SoccerPitchConfiguration()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Function to mirror y-coordinates
    def mirror_y(y):
        return pitch.width - y  # Mirror along the horizontal axis (center of the pitch)
    
    # Draw edges
    for edge in pitch.edges:
        p1, p2 = pitch.vertices[edge[0] - 1], pitch.vertices[edge[1] - 1]
        # Mirror y-coordinates
        p1_mirrored = (p1[0], mirror_y(p1[1]))
        p2_mirrored = (p2[0], mirror_y(p2[1]))
        x_values = [p1_mirrored[0], p2_mirrored[0]]
        y_values = [p1_mirrored[1], p2_mirrored[1]]
        ax.plot(x_values, y_values, 'k-', linewidth=1.5, alpha=0.7)

    # Draw center circle (mirror its y-coordinate)
    centre_circle_mirrored = (pitch.length / 2, mirror_y(pitch.width / 2))
    centre_circle = plt.Circle(centre_circle_mirrored, pitch.centre_circle_radius, color='black', fill=False, linewidth=1.5, alpha=0.7)
    ax.add_patch(centre_circle)

    # Draw penalty spots (mirror their y-coordinates)
    penalty_spot_left_mirrored = (pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    penalty_spot_right_mirrored = (pitch.length - pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    ax.scatter([penalty_spot_left_mirrored[0], penalty_spot_right_mirrored[0]], 
               [penalty_spot_left_mirrored[1], penalty_spot_right_mirrored[1]], color='black', s=20, alpha=0.7, zorder=3)

    # Extract x and y coordinates for the heatmap
    x_positions = [pos[0] for pos in positions]
    y_positions = [mirror_y(pos[1]) for pos in positions]  # Mirror y-coordinates
    
    # Create custom heatmap using kernel density estimation
    # Define the grid on which to calculate the KDE
    x_grid = np.linspace(0, pitch.length, 100)
    y_grid = np.linspace(0, pitch.width, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions_grid = np.vstack([X.ravel(), Y.ravel()])
    
    # Calculate the kernel density estimate
    from scipy.stats import gaussian_kde
    if len(x_positions) > 1:  # Ensure we have enough points for KDE
        kernel = gaussian_kde(np.vstack([x_positions, y_positions]))
        Z = np.reshape(kernel(positions_grid), X.shape)
        
        # Create a custom colormap from green to red
        colors = [(0, 0.5, 0), (1, 1, 0), (1, 0, 0)]  # Green -> Yellow -> Red
        cmap_name = 'green_yellow_red'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
        
        # Plot the kernel density estimate
        heatmap = ax.imshow(Z, cmap=cm, extent=[0, pitch.length, 0, pitch.width], 
                           origin='lower', alpha=0.7, aspect='auto')
        
        # Add a colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Ball Position Density')
    else:
        # If there are not enough points for KDE, just show the scatter plot
        ax.scatter(x_positions, y_positions, color='blue', s=15, alpha=0.7)

    # Set limits and aspect ratio
    ax.set_xlim(0, pitch.length)
    ax.set_ylim(0, pitch.width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_title("Soccer Pitch Ball Position Heatmap")

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig("soccer_pitch_ball_heatmap.png", dpi=300)
    print("Heatmap saved to soccer_pitch_ball_heatmap.png")

# Load and process the ball positions
with open("tracks_pickles/ball_positions.pickle", "rb") as f:
    positions = pickle.load(f)

# Generate the heatmap
draw_pitch_with_heatmap(positions["transformed_position"])

# Optional: Also create a simple scatter plot version for comparison
def draw_pitch_with_scatter(positions):
    pitch = SoccerPitchConfiguration()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Function to mirror y-coordinates
    def mirror_y(y):
        return pitch.width - y
    
    # Draw edges
    for edge in pitch.edges:
        p1, p2 = pitch.vertices[edge[0] - 1], pitch.vertices[edge[1] - 1]
        p1_mirrored = (p1[0], mirror_y(p1[1]))
        p2_mirrored = (p2[0], mirror_y(p2[1]))
        x_values = [p1_mirrored[0], p2_mirrored[0]]
        y_values = [p1_mirrored[1], p2_mirrored[1]]
        ax.plot(x_values, y_values, 'k-', linewidth=1.5)

    # Draw center circle
    centre_circle_mirrored = (pitch.length / 2, mirror_y(pitch.width / 2))
    centre_circle = plt.Circle(centre_circle_mirrored, pitch.centre_circle_radius, color='black', fill=False, linewidth=1.5)
    ax.add_patch(centre_circle)

    # Draw penalty spots
    penalty_spot_left_mirrored = (pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    penalty_spot_right_mirrored = (pitch.length - pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    ax.scatter([penalty_spot_left_mirrored[0], penalty_spot_right_mirrored[0]], 
               [penalty_spot_left_mirrored[1], penalty_spot_right_mirrored[1]], color='black', s=20, zorder=3)

    # Draw ball positions with transparency for density visualization
    x_positions = [pos[0] for pos in positions]
    y_positions = [mirror_y(pos[1]) for pos in positions]
    ax.scatter(x_positions, y_positions, color='blue', s=15, alpha=0.5)

    ax.set_xlim(0, pitch.length)
    ax.set_ylim(0, pitch.width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_title("Soccer Pitch with Ball Positions (Scatter)")

    plt.savefig("soccer_pitch_ball_scatter.png", dpi=300)
    print("Scatter plot saved to soccer_pitch_ball_scatter.png")

# Uncomment to also generate a scatter plot
# draw_pitch_with_scatter(positions["transformed_position"])