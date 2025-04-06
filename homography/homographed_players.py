import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import os
from scipy.signal import savgol_filter
from collections import defaultdict

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

def draw_pitch(ax, vertical_flip=True):
    """
    Draw a soccer pitch on the given axes with option to flip vertically
    
    Args:
        ax: Matplotlib axes to draw on
        vertical_flip: Whether to flip the pitch vertically
    """
    pitch = SoccerPitchConfiguration()
    
    # Draw edges/lines
    for edge in pitch.edges:
        p1, p2 = pitch.vertices[edge[0] - 1], pitch.vertices[edge[1] - 1]
        x_values = [p1[0], p2[0]]
        
        if vertical_flip:
            # Flip y-coordinates
            y_values = [pitch.width - p1[1], pitch.width - p2[1]]
        else:
            y_values = [p1[1], p2[1]]
            
        ax.plot(x_values, y_values, 'white', linewidth=1.5, alpha=0.7)

    # Draw center circle
    if vertical_flip:
        centre_y = pitch.width - pitch.width / 2  # Flip the y-coordinate
    else:
        centre_y = pitch.width / 2
        
    centre_circle = plt.Circle((pitch.length / 2, centre_y), 
                              pitch.centre_circle_radius, 
                              color='white', fill=False, linewidth=1.5, alpha=0.7)
    ax.add_patch(centre_circle)

    # Draw center line
    if vertical_flip:
        y_values = [pitch.width, 0]  # Flipped y-coordinates
    else:
        y_values = [0, pitch.width]
        
    ax.plot([pitch.length/2, pitch.length/2], y_values, 'white', linewidth=1.5, alpha=0.7)

    # Draw penalty spots
    penalty_spot_left = (pitch.penalty_spot_distance, pitch.width / 2)
    penalty_spot_right = (pitch.length - pitch.penalty_spot_distance, pitch.width / 2)
    
    if vertical_flip:
        # Flip y-coordinates
        penalty_spot_y = pitch.width - pitch.width / 2
    else:
        penalty_spot_y = pitch.width / 2
        
    ax.scatter([penalty_spot_left[0], penalty_spot_right[0]], 
               [penalty_spot_y, penalty_spot_y], 
               color='white', s=20, alpha=0.7, zorder=3)

    # Draw penalty arcs
    if vertical_flip:
        # Flip the y-coordinate
        arc_center_y = pitch.width - pitch.width / 2
        # Also need to adjust the theta angles for the flipped arcs
        left_arc_theta1, left_arc_theta2 = 310, 50
        right_arc_theta1, right_arc_theta2 = 130, 230
    else:
        arc_center_y = pitch.width / 2
        left_arc_theta1, left_arc_theta2 = 310, 50
        right_arc_theta1, right_arc_theta2 = 130, 230
    
    # Left penalty arc
    penalty_arc_left = patches.Arc((pitch.penalty_spot_distance, arc_center_y),
                                  2 * pitch.centre_circle_radius, 2 * pitch.centre_circle_radius,
                                  theta1=left_arc_theta1, theta2=left_arc_theta2, 
                                  angle=0.0, linewidth=1.5, color='white', alpha=0.7)
    ax.add_patch(penalty_arc_left)
    
    # Right penalty arc
    penalty_arc_right = patches.Arc((pitch.length - pitch.penalty_spot_distance, arc_center_y),
                                   2 * pitch.centre_circle_radius, 2 * pitch.centre_circle_radius,
                                   theta1=right_arc_theta1, theta2=right_arc_theta2, 
                                   angle=0.0, linewidth=1.5, color='white', alpha=0.7)
    ax.add_patch(penalty_arc_right)
    
    # Set pitch background color
    ax.set_facecolor('#1a6638')  # Dark green color
    
    # Set limits and aspect ratio
    ax.set_xlim(-500, pitch.length + 500)
    ax.set_ylim(-500, pitch.width + 500)
    ax.set_aspect('equal')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    return pitch

def filter_outliers(homographed_data, pitch_bounds_factor=1.2):
    """
    Filter outliers that are far outside the pitch boundaries
    
    Args:
        homographed_data: List of frames with player positions
        pitch_bounds_factor: Factor to multiply pitch dimensions for outlier detection
        
    Returns:
        The filtered homographed data
    """
    print(f"Filtering outliers...")
    
    # Create a copy to avoid modifying the original data
    filtered_data = []
    
    # Get pitch dimensions
    pitch = SoccerPitchConfiguration()
    min_x, max_x = -500, pitch.length + 500
    min_y, max_y = -500, pitch.width + 500
    
    # Extend the boundaries by the factor for outlier detection
    extended_min_x = min_x * pitch_bounds_factor
    extended_max_x = max_x * pitch_bounds_factor
    extended_min_y = min_y * pitch_bounds_factor
    extended_max_y = max_y * pitch_bounds_factor
    
    outlier_count = 0
    total_positions = 0
    
    # Filter each frame
    for frame_data in homographed_data:
        filtered_frame = {}
        for player_id, player_data in frame_data.items():
            total_positions += 1
            if isinstance(player_data, dict) and 'pitch_position' in player_data:
                x, y = player_data['pitch_position']
                
                # Check if the position is within the extended boundaries
                if (extended_min_x <= x <= extended_max_x) and (extended_min_y <= y <= extended_max_y):
                    filtered_frame[player_id] = player_data
                else:
                    outlier_count += 1
        
        filtered_data.append(filtered_frame)
    
    print(f"Removed {outlier_count} outliers out of {total_positions} positions ({outlier_count/total_positions*100:.1f}%)")
    
    return filtered_data

def smooth_trajectories(homographed_data, window_size=21, poly_order=2):
    """
    Apply stronger trajectory smoothing to reduce bouncing effect
    
    Args:
        homographed_data: List of frames with player positions
        window_size: Size of the smoothing window (odd number)
        poly_order: Order of the polynomial to fit
    
    Returns:
        The smoothed homographed data
    """
    print(f"Applying trajectory smoothing (window_size={window_size}, poly_order={poly_order})...")
    
    # First, organize data by player
    player_trajectories = defaultdict(list)
    
    for frame_idx, frame_data in enumerate(homographed_data):
        for player_id, player_data in frame_data.items():
            if isinstance(player_data, dict) and 'pitch_position' in player_data:
                # Store position with frame index for later reconstruction
                player_trajectories[player_id].append({
                    'frame_idx': frame_idx,
                    'position': player_data['pitch_position'],
                    'data': player_data  # Store the full player data
                })
    
    # Create a new list to hold the smoothed data
    smoothed_data = [{} for _ in range(len(homographed_data))]
    
    # For each player, smooth their trajectory
    for player_id, trajectory in player_trajectories.items():
        # Skip if too few points
        if len(trajectory) < window_size:
            print(f"Warning: Not enough points to smooth trajectory for player {player_id} ({len(trajectory)} points)")
            for point in trajectory:
                frame_idx = point['frame_idx']
                smoothed_data[frame_idx][player_id] = point['data']
            continue
        
        # Sort by frame index
        trajectory.sort(key=lambda x: x['frame_idx'])
        
        # Extract frame indices and positions
        frames = [point['frame_idx'] for point in trajectory]
        x_positions = [point['position'][0] for point in trajectory]
        y_positions = [point['position'][1] for point in trajectory]
        
        try:
            # Apply Savitzky-Golay filter with stronger smoothing
            x_smooth = savgol_filter(x_positions, window_size, poly_order)
            y_smooth = savgol_filter(y_positions, window_size, poly_order)
            
            # Apply additional median filtering to remove spikes
            from scipy.signal import medfilt
            x_smooth = medfilt(x_smooth, kernel_size=5)
            y_smooth = medfilt(y_smooth, kernel_size=5)
            
            # Reconstruct the trajectory with smoothed positions
            for i, point in enumerate(trajectory):
                frame_idx = point['frame_idx']
                # Copy the original data but update the position
                smoothed_player_data = point['data'].copy()
                smoothed_player_data['pitch_position'] = [x_smooth[i], y_smooth[i]]
                
                # Apply vertical flip to y-coordinate if needed
                # This is handled elsewhere when drawing
                
                smoothed_data[frame_idx][player_id] = smoothed_player_data
                
        except Exception as e:
            print(f"Warning: Smoothing failed for player {player_id}: {e}")
            # Fall back to original data if smoothing fails
            for point in trajectory:
                frame_idx = point['frame_idx']
                smoothed_data[frame_idx][player_id] = point['data']
    
    return smoothed_data

def load_team_colors_and_players():
    """
    Load team colors and player team assignments from pickle files
    
    Returns:
        tuple: (team_colors, player_teams) where:
            - team_colors is a dict mapping team_id to RGB color values
            - player_teams is a dict mapping player_id to team_id
    """
    try:
        # Load team colors
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/teams.pkl", "rb") as f:
            team_colors = pickle.load(f)
        
        # Load player team assignments
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl", "rb") as f:
            player_teams = pickle.load(f)
        
        print("Team colors and player assignments loaded successfully")
        
        # Convert np.int32 to regular Python integers if needed
        player_teams_cleaned = {}
        for player_id, team_id in player_teams.items():
            if hasattr(team_id, 'item'):  # For numpy integers
                player_teams_cleaned[player_id] = team_id.item()
            else:
                player_teams_cleaned[player_id] = team_id
        
        # Convert BGR to RGB for matplotlib if necessary
        team_colors_rgb = {}
        for team_id, color in team_colors.items():
            # Normalize to 0-1 range for matplotlib
            if isinstance(color, tuple) and len(color) == 3:
                team_colors_rgb[team_id] = tuple(c/255 for c in color)
            else:
                team_colors_rgb[team_id] = color
        
        return team_colors_rgb, player_teams_cleaned
        
    except Exception as e:
        print(f"Error loading team data: {e}")
        # Return default colors if files not found
        default_colors = {
            0: (0, 0, 1),  # Blue
            1: (1, 0, 0),  # Red
            -1: (0, 1, 0)  # Green for goalkeepers
        }
        return default_colors, {}

def get_team_id_for_player(player_id, player_teams):
    """Get the team ID for a player from the player_teams dictionary"""
    player_id_str = str(player_id)
    if player_id_str in player_teams:
        return player_teams[player_id_str]
    
    # If player ID not found, try to infer based on ID number
    try:
        player_id_int = int(player_id)
        return 0 if player_id_int % 2 == 0 else 1
    except (ValueError, TypeError):
        return None

def draw_homographed_players_with_teams(homographed_data, frame_idx=None, output_path=None, vertical_flip=True):
    """
    Draw homographed players on a soccer pitch with team colors from pickle files
    
    Args:
        homographed_data: List of frames with player positions
        frame_idx: If provided, only draw the specified frame, otherwise draw all frames
        output_path: If provided, save the figure to this path
        vertical_flip: Whether to flip the pitch and players vertically
    """
    # Load team colors and player assignments
    team_colors, player_teams = load_team_colors_and_players()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('black')
    
    # Draw the pitch
    pitch = draw_pitch(ax, vertical_flip=vertical_flip)
    
    # Set default colors for any team IDs not in team_colors
    for team_id in [0, 1, -1, None]:
        if team_id not in team_colors:
            if team_id == 0:
                team_colors[team_id] = (0, 0, 1)  # Blue
            elif team_id == 1:
                team_colors[team_id] = (1, 0, 0)  # Red
            elif team_id == -1:
                team_colors[team_id] = (0, 1, 0)  # Green for goalkeepers
            else:
                team_colors[team_id] = (0.5, 0.5, 0.5)  # Gray for unknown
    
    marker_size = 120
    alpha = 0.7
    
    if frame_idx is not None and frame_idx < len(homographed_data):
        # Draw only the specified frame
        frame_data = homographed_data[frame_idx]
        
        # Plot players
        for player_id, player_data in frame_data.items():
            if isinstance(player_data, dict) and 'pitch_position' in player_data:
                x, y = player_data['pitch_position']
                
                # Apply vertical flip if needed
                if vertical_flip:
                    y = pitch.width - y
                
                # Get team ID from player_data if available, otherwise from player_teams
                team_id = player_data.get('team_id', None)
                if team_id is None:
                    team_id = get_team_id_for_player(player_id, player_teams)
                
                # Get color for this team
                color = team_colors.get(team_id, (0.5, 0.5, 0.5))  # Gray for unknown team
                
                # Draw player
                ax.scatter(x, y, s=marker_size, color=color, alpha=alpha, edgecolors='white', linewidths=1)
                
                # Add player ID
                ax.text(x, y+100, str(player_id), color='white', fontsize=8, 
                        ha='center', va='center', bbox=dict(facecolor=color, alpha=0.5, boxstyle='round,pad=0.2'))
        
        ax.set_title(f"Frame {frame_idx}", color='white', fontsize=14)
    else:
        # Draw all frames with faded positions (heatmap style)
        team_positions = {0: [], 1: [], -1: [], None: []}
        
        # Collect all positions across all frames
        for frame_data in homographed_data:
            for player_id, player_data in frame_data.items():
                if isinstance(player_data, dict) and 'pitch_position' in player_data:
                    x, y = player_data['pitch_position']
                    
                    # Apply vertical flip if needed
                    if vertical_flip:
                        y = pitch.width - y
                    
                    # Get team ID from player_data if available, otherwise from player_teams
                    team_id = player_data.get('team_id', None)
                    if team_id is None:
                        team_id = get_team_id_for_player(player_id, player_teams)
                    
                    team_positions[team_id].append((x, y))
        
        # Plot heatmap for each team
        for team_id, positions in team_positions.items():
            if positions:
                x_pos = [pos[0] for pos in positions]
                y_pos = [pos[1] for pos in positions]
                
                # Use a smaller point size for the heatmap
                color = team_colors.get(team_id, (0.5, 0.5, 0.5))  # Gray for unknown team
                ax.scatter(x_pos, y_pos, s=marker_size/3, color=color, 
                           alpha=0.2, edgecolors=None)
        
        ax.set_title("All Frames - Player Positions", color='white', fontsize=14)
    
    # Add legend with actual team colors
    legend_elements = []
    for team_id, color in team_colors.items():
        if team_id == 0:
            label = "Team 1"
        elif team_id == 1:
            label = "Team 2"
        elif team_id == -1:
            label = "Goalkeepers"
        else:
            label = f"Team {team_id}"
            
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        )
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig, ax

def create_animation_with_teams(homographed_data, output_file='player_movement_teams.mp4', frames_to_use=None, fps=10, vertical_flip=True):
    """
    Create an animation of the players moving on the pitch with team colors from pickle files
    
    Args:
        homographed_data: List of frames with player positions
        output_file: Path to save the animation
        frames_to_use: List of frame indices to include in the animation
        fps: Frames per second for the animation
        vertical_flip: Whether to flip the pitch and players vertically
    """
    # Load team colors and player assignments
    team_colors, player_teams = load_team_colors_and_players()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('black')
    
    # Draw the pitch
    pitch = draw_pitch(ax, vertical_flip=vertical_flip)
    
    # Set default colors for any team IDs not in team_colors
    for team_id in [0, 1, -1, None]:
        if team_id not in team_colors:
            if team_id == 0:
                team_colors[team_id] = (0, 0, 1)  # Blue
            elif team_id == 1:
                team_colors[team_id] = (1, 0, 0)  # Red
            elif team_id == -1:
                team_colors[team_id] = (0, 1, 0)  # Green for goalkeepers
            else:
                team_colors[team_id] = (0.5, 0.5, 0.5)  # Gray for unknown
    
    # Get the list of frames to use
    if frames_to_use is None:
        frames_to_use = list(range(len(homographed_data)))
    else:
        frames_to_use = [f for f in frames_to_use if f < len(homographed_data)]
    
    print(f"Creating animation with {len(frames_to_use)} frames...")
    
    # Set up a scatter plot for each team
    scatter_plots = {}
    for team_id, color in team_colors.items():
        scatter_plots[team_id] = ax.scatter([], [], s=120, color=color, alpha=0.7, edgecolors='white', linewidths=1)
    
    # Add default scatter plot for any team not in team_colors
    for team_id in [0, 1, -1, None]:
        if team_id not in scatter_plots:
            scatter_plots[team_id] = ax.scatter([], [], s=120, color=(0.5, 0.5, 0.5), alpha=0.7, edgecolors='white', linewidths=1)
    
    # Set up a title
    title = ax.set_title("Frame: 0", color='white', fontsize=14)
    
    # Add ball visualization if present in the data
    ball_scatter = ax.scatter([], [], s=80, color='white', alpha=1.0, edgecolors='black', linewidths=1, zorder=10)
    
    # Add legend with actual team colors
    legend_elements = []
    for team_id, color in team_colors.items():
        if team_id == 0:
            label = "Team 1"
        elif team_id == 1:
            label = "Team 2"
        elif team_id == -1:
            label = "Goalkeepers"
        else:
            label = f"Team {team_id}"
            
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        )
    
    # Add ball to legend
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, label='Ball')
    )
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
    
    # Text annotations for player IDs - we'll update these in the animation function
    player_labels = {}
    
    def init():
        for scatter in scatter_plots.values():
            scatter.set_offsets(np.empty((0, 2)))
        ball_scatter.set_offsets(np.empty((0, 2)))
        title.set_text("Frame: 0")
        # Clear any existing text annotations
        for label in player_labels.values():
            label.remove()
        player_labels.clear()
        return list(scatter_plots.values()) + [ball_scatter, title]
    
    def update(i):
        frame_idx = frames_to_use[i]
        frame_data = homographed_data[frame_idx]
        
        # Organize players by team
        team_positions = defaultdict(list)
        team_player_ids = defaultdict(list)
        ball_pos = []
        
        for player_id, player_data in frame_data.items():
            if isinstance(player_data, dict) and 'pitch_position' in player_data:
                x, y = player_data['pitch_position']
                
                # Apply vertical flip if needed
                if vertical_flip:
                    y = pitch.width - y
                
                # Check if this is the ball (assuming ball has player_id "ball" or similar)
                if str(player_id).lower() == "ball":
                    ball_pos = [[x, y]]
                    continue
                    
                # Get team ID from player_data if available, otherwise from player_teams
                team_id = player_data.get('team_id', None)
                if team_id is None:
                    team_id = get_team_id_for_player(player_id, player_teams)
                
                # Add to team positions
                team_positions[team_id].append([x, y])
                team_player_ids[team_id].append(player_id)
        
        # Update scatter plots
        for team_id, positions in team_positions.items():
            if team_id in scatter_plots:
                if positions:
                    scatter_plots[team_id].set_offsets(positions)
                else:
                    scatter_plots[team_id].set_offsets(np.empty((0, 2)))
        
        # Update ball position
        if ball_pos:
            ball_scatter.set_offsets(ball_pos)
        else:
            ball_scatter.set_offsets(np.empty((0, 2)))
        
        # Clear old annotations
        for label in player_labels.values():
            label.remove()
        player_labels.clear()
        
        # Add new annotations
        for team_id, positions in team_positions.items():
            player_ids = team_player_ids[team_id]
            color = team_colors.get(team_id, (0.5, 0.5, 0.5))  # Gray for unknown team
            
            for i, (x, y) in enumerate(positions):
                player_id = player_ids[i]
                label = ax.text(x, y+100, str(player_id), color='white', fontsize=8, 
                               ha='center', va='center', bbox=dict(facecolor=color, alpha=0.5, boxstyle='round,pad=0.2'))
                player_labels[f"{team_id}_{player_id}"] = label
        
        title.set_text(f"Frame: {frame_idx}")
        
        return list(scatter_plots.values()) + [ball_scatter, title] + list(player_labels.values())
    
    # Create animation - use all frames
    anim = animation.FuncAnimation(fig, update, frames=len(frames_to_use), 
                                  init_func=init, blit=True, interval=1000/fps)
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    
    try:
        # Try with ffmpeg
        anim.save(output_file, writer='ffmpeg', fps=fps, dpi=150, bitrate=2500)
    except Exception as e:
        print(f"Failed to save with ffmpeg: {e}")
        # Fall back to other writers
        try:
            # Try with pillow
            anim.save(output_file, writer='pillow', fps=fps, dpi=100)
        except Exception as e:
            print(f"Failed to save with pillow: {e}")
            # If all else fails, try a different format
            alt_output = os.path.splitext(output_file)[0] + '.gif'
            print(f"Trying to save as GIF to {alt_output}...")
            anim.save(alt_output, writer='pillow', fps=fps//2, dpi=80)  # Lower quality for GIF
    
    print(f"Animation saved to {output_file}")
    
    return anim

def filter_players_by_team(homographed_data, team_id, player_teams=None):
    """
    Filter homographed data to show only players of a specific team
    
    Args:
        homographed_data: List of frames with player positions
        team_id: Team ID to filter for (0, 1, or -1 for goalkeepers)
        player_teams: Dictionary mapping player_id to team_id (if None, will be loaded)
    
    Returns:
        Filtered homographed data with only the specified team
    """
    if player_teams is None:
        _, player_teams = load_team_colors_and_players()
    
    # Create a new list to hold the filtered data
    filtered_data = []
    
    # Filter each frame
    for frame_data in homographed_data:
        filtered_frame = {}
        
        for player_id, player_data in frame_data.items():
            if isinstance(player_data, dict) and 'pitch_position' in player_data:
                # Get team ID from player_data if available, otherwise from player_teams
                player_team_id = player_data.get('team_id', None)
                if player_team_id is None:
                    player_team_id = get_team_id_for_player(player_id, player_teams)
                
                # Include if team matches or if it's the ball
                if player_team_id == team_id or str(player_id).lower() == "ball":
                    filtered_frame[player_id] = player_data
        
        filtered_data.append(filtered_frame)
    
    return filtered_data

def main_with_teams_improved():
    """
    Improved main function with better smoothing and faster animation generation
    """
    import os
    import pickle
    import time
    
    start_time = time.time()
    
    # Try multiple possible paths for the homographed data
    possible_paths = [
        "/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/homographed_tracks.pkl",
        "/home/elmehdi/Desktop/footballe_analyseur/homography/homographed_players.pkl",
        "homographed_players.pkl",
        "../homographed_players.pkl",
    ]
    
    # Find the first path that exists
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("Could not find homographed data file in any of the expected locations.")
        data_path = input("Please enter the full path to the homographed data file: ")
    
    # Load the homographed data
    try:
        print(f"Loading homographed data from {data_path}...")
        with open(data_path, "rb") as f:
            homographed_data = pickle.load(f)
        
        print(f"Loaded {len(homographed_data)} frames of homographed data")
    except Exception as e:
        print(f"Error loading homographed data: {e}")
        return
    
    # Filter outliers more aggressively
    homographed_data = filter_outliers(homographed_data, pitch_bounds_factor=1.1)
    
    # Apply much stronger smoothing to reduce bouncing
    # Increase window size and use lower poly order for smoother curves
    homographed_data = smooth_trajectories(homographed_data, window_size=31, poly_order=1)
    
    # Apply a second pass of smoothing for even smoother trajectories
    homographed_data = smooth_trajectories(homographed_data, window_size=21, poly_order=1)
    
    # Create output directory if it doesn't exist
    output_dir = "/home/elmehdi/Desktop/footballe_analyseur/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the team colors and player assignments
    team_colors, player_teams = load_team_colors_and_players()
    
    # Generate an overall visualization with all frames (heatmap)
    output_path = os.path.join(output_dir, "all_frames_visualization.png")
    draw_homographed_players_with_teams(
        homographed_data, 
        frame_idx=None,  # None means visualize all frames
        output_path=output_path, 
        vertical_flip=True
    )
    print(f"Overall visualization saved to {output_path}")
    
    # Create a faster animation by sampling frames
    print("Creating optimized animation (this should be faster)...")
    
    # For faster processing, use 1/4 of the frames 
    # This will make the generation faster while still maintaining visual smoothness
    total_frames = len(homographed_data)
    frame_step = 4  # Only use every 4th frame
    frames_to_use = list(range(0, total_frames, frame_step))
    
    # Make sure we include at least 100 frames for a decent animation
    if len(frames_to_use) < 100:
        frame_step = max(1, total_frames // 100)
        frames_to_use = list(range(0, total_frames, frame_step))
    
    output_path = os.path.join(output_dir, "player_movement_optimized.mp4")
    create_animation_with_teams(
        homographed_data,
        output_file=output_path,
        frames_to_use=frames_to_use,
        fps=20,  # Higher FPS for smoother playback despite fewer frames
        vertical_flip=True
    )
    print(f"Optimized animation saved to {output_path}")
    
    elapsed_time = time.time() - start_time
    print(f"All visualizations complete! Total processing time: {elapsed_time:.1f} seconds")

# You can use this function to apply more aggressive smoothing to already processed data
def extra_smooth_trajectories(homographed_data):
    """
    Apply extra strong smoothing to further reduce player bouncing
    """
    print("Applying extra smoothing to reduce bouncing...")
    
    # First smoothing pass with large window
    smoothed_data = smooth_trajectories(homographed_data, window_size=31, poly_order=1)
    
    # Second smoothing pass with slightly smaller window
    smoothed_data = smooth_trajectories(smoothed_data, window_size=21, poly_order=1)
    
    # Third pass with median filtering to remove any remaining spikes
    from scipy.signal import medfilt
    
    # Organize by player
    player_trajectories = defaultdict(list)
    for frame_idx, frame_data in enumerate(smoothed_data):
        for player_id, player_data in frame_data.items():
            if isinstance(player_data, dict) and 'pitch_position' in player_data:
                player_trajectories[player_id].append({
                    'frame_idx': frame_idx,
                    'position': player_data['pitch_position'],
                    'data': player_data
                })
    
    # Apply median filtering to each player trajectory
    extra_smooth_data = [{} for _ in range(len(smoothed_data))]
    for player_id, trajectory in player_trajectories.items():
        if len(trajectory) > 7:  # Need reasonable amount of data for median filter
            trajectory.sort(key=lambda x: x['frame_idx'])
            
            # Extract positions
            x_positions = [point['position'][0] for point in trajectory]
            y_positions = [point['position'][1] for point in trajectory]
            
            # Apply median filter
            try:
                x_smooth = medfilt(x_positions, kernel_size=7)
                y_smooth = medfilt(y_positions, kernel_size=7)
                
                # Reconstruct trajectory
                for i, point in enumerate(trajectory):
                    frame_idx = point['frame_idx']
                    new_data = point['data'].copy()
                    new_data['pitch_position'] = [x_smooth[i], y_smooth[i]]
                    extra_smooth_data[frame_idx][player_id] = new_data
            except Exception as e:
                print(f"Warning: Extra smoothing failed for player {player_id}: {e}")
                for point in trajectory:
                    frame_idx = point['frame_idx']
                    extra_smooth_data[frame_idx][player_id] = point['data']
        else:
            # Not enough data for median filtering
            for point in trajectory:
                frame_idx = point['frame_idx']
                extra_smooth_data[frame_idx][player_id] = point['data']
    
    return extra_smooth_data

# Replace the main function call with this improved version
if __name__ == "__main__":
    main_with_teams_improved()