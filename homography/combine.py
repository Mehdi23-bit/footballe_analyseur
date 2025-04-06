import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.patches import Rectangle, Arc
import os
import matplotlib.animation as animation
from matplotlib.lines import Line2D

def load_team_data():
    """Load team colors and player assignments from pickle files"""
    try:
        # Load team colors
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/teams.pkl", "rb") as f:
            teams = pickle.load(f)
        
        # Load player team assignments
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl", "rb") as f:
            players = pickle.load(f)
        
        print("Team colors:", teams)
        print(f"Loaded {len(players)} player team assignments")
        
        # Convert colors to matplotlib format (0-1 range)
        team_colors = {}
        for team_id, color in teams.items():
            if isinstance(color, tuple) and len(color) == 3:
                # Convert to RGB in 0-1 range
                team_colors[team_id] = tuple(c/255 for c in color)
            else:
                team_colors[team_id] = color
        
        # Convert numpy integers to Python integers
        player_teams = {}
        for player_id, team_id in players.items():
            if hasattr(team_id, 'item'):
                player_teams[player_id] = team_id.item()
            else:
                player_teams[player_id] = team_id
        
        # Add default colors for any missing team IDs
        if -1 not in team_colors:
            team_colors[-1] = (0, 1, 0)  # Green for goalkeepers
        
        if None not in team_colors:
            team_colors[None] = (0.5, 0.5, 0.5)  # Gray for unknown
            
        return team_colors, player_teams
    
    except Exception as e:
        print(f"Error loading team data: {e}")
        # Return default colors
        return {0: (0.9, 0.9, 0.86), 1: (0.58, 0.91, 0.69), -1: (0, 1, 0), None: (0.5, 0.5, 0.5)}, {}

def draw_soccer_pitch(ax, vertical_flip=True):
    """
    Draw a simple soccer pitch on the given axes
    """
    # Set pitch dimensions
    pitch_length = 12000
    pitch_width = 7000
    
    # Set background color
    ax.set_facecolor('#1a6638')  # Dark green
    
    # Draw pitch outline (using white lines)
    ax.plot([0, 0, pitch_length, pitch_length, 0], 
            [0, pitch_width, pitch_width, 0, 0], 'white', lw=2)
    
    # Draw halfway line
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], 'white', lw=2)
    
    # Draw center circle
    center_circle = plt.Circle((pitch_length/2, pitch_width/2), 915, 
                              color='white', fill=False, lw=2)
    ax.add_patch(center_circle)
    
    # Set limits and remove axis ticks
    ax.set_xlim(-500, pitch_length + 500)
    ax.set_ylim(-500, pitch_width + 500)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    return pitch_length, pitch_width

def visualize_players_and_ball(players_data_path, ball_data_path, output_path=None, 
                             vertical_flip=True, excluded_players=None, fps=25):
    """
    Create an animation of players and ball together
    
    Args:
        players_data_path: Path to the pickle file with player data
        ball_data_path: Path to the pickle file with ball data
        output_path: Path to save the animation
        vertical_flip: Whether to flip the pitch vertically
        excluded_players: List of player IDs to exclude from visualization
        fps: Frames per second for the animation
    """
    # Define players to exclude
    if excluded_players is None:
        excluded_players = ["21", "20", "3", "105", "127", "121"]
    
    # Convert to strings if needed
    excluded_players = [str(p) for p in excluded_players]
    
    print(f"Excluding players: {', '.join(excluded_players)}")
    
    # Load team data
    team_colors, player_teams = load_team_data()
    
    try:
        # Load the player position data
        print(f"Loading player position data from {players_data_path}...")
        with open(players_data_path, "rb") as f:
            player_frames = pickle.load(f)
        
        if not isinstance(player_frames, list) or len(player_frames) == 0:
            print("Invalid player data format or empty data")
            return
            
        total_player_frames = len(player_frames)
        print(f"Loaded {total_player_frames} frames of player position data")
        
        # Load the ball position data
        print(f"Loading ball position data from {ball_data_path}...")
        with open(ball_data_path, "rb") as f:
            ball_positions = pickle.load(f)
        
        print(f"Loaded {len(ball_positions)} frames of ball position data")
        
        # Create figure and axes for animation
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.set_facecolor('black')
        
        # Draw the pitch - this stays the same for all frames
        pitch_length, pitch_width = draw_soccer_pitch(ax, vertical_flip)
        
        # Add legend for teams
        legend_elements = []
        for team_id, color in team_colors.items():
            if team_id == 0:
                label = "Team 1"
            elif team_id == 1:
                label = "Team 2"
            elif team_id == -1:
                label = "Goalkeepers"
            else:
                label = "Unknown Team"
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color, markersize=10, label=label))
        
        # Add ball to legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='white', markersize=8, label='Ball'))
        
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
        
        # Set up elements that will be updated in the animation
        title = ax.set_title(f"Frame: 0", color='white', fontsize=14)
        
        # Create scatter plot objects for each team - we'll update these
        scatter_objects = {}
        for team_id, color in team_colors.items():
            scatter_objects[team_id] = ax.scatter([], [], s=150, color=color, edgecolors='white', linewidths=1, alpha=0.8)
        
        # Create a scatter plot for the ball
        ball_scatter = ax.scatter([], [], s=100, color='white', edgecolors='black', linewidths=1, zorder=10)
        
        # Create a line to show the ball's trail
        trail_length = 20  # Number of previous positions to show
        ball_trail, = ax.plot([], [], 'yellow', alpha=0.5, linewidth=2, zorder=9)
        
        # Create a dictionary to store player ID text objects
        player_labels = {}
        
        # Store the last valid ball position for the trail
        last_valid_trail_x = []
        last_valid_trail_y = []
        
        def init():
            """Initialize animation"""
            for scatter in scatter_objects.values():
                scatter.set_offsets(np.empty((0, 2)))
            
            ball_scatter.set_offsets(np.empty((0, 2)))
            ball_trail.set_data([], [])
            
            title.set_text("Frame: 0")
            
            # Clear any existing player labels
            for label in player_labels.values():
                label.remove()
            player_labels.clear()
            
            return [title, ball_scatter, ball_trail] + list(scatter_objects.values())
        
        def update(frame_idx):
            """Update function for animation"""
            nonlocal last_valid_trail_x, last_valid_trail_y
            
            # Update the frame title
            title.set_text(f"Frame: {frame_idx}")
            
            # Get the player frame data
            if frame_idx < len(player_frames):
                player_frame = player_frames[frame_idx]
                
                # Organize player positions by team
                team_positions = {0: [], 1: [], -1: [], None: []}
                team_player_ids = {0: [], 1: [], -1: [], None: []}
                
                # Clear old player labels
                for label in player_labels.values():
                    label.remove()
                player_labels.clear()
                
                # Process each player in the frame
                for player_id, player_info in player_frame.items():
                    if isinstance(player_info, dict) and 'pitch_position' in player_info:
                        # Skip excluded players
                        if str(player_id) in excluded_players:
                            continue
                        
                        # Skip players not in our player_teams dictionary
                        if str(player_id) not in player_teams and player_id != "ball":
                            continue
                        
                        # Get position
                        x, y = player_info['pitch_position']
                        
                        # Vertical flip if requested
                        if vertical_flip:
                            y = pitch_width - y
                        
                        # Check if it's the ball (should be handled separately now)
                        if str(player_id).lower() == "ball":
                            continue
                        
                        # Get team ID from our player assignments
                        team_id = player_teams.get(str(player_id), None)
                        
                        # Add to the right team's data
                        team_positions[team_id].append([x, y])
                        team_player_ids[team_id].append(player_id)
                
                # Update scatter plots for each team
                for team_id, positions in team_positions.items():
                    if positions:
                        scatter_objects[team_id].set_offsets(positions)
                    else:
                        scatter_objects[team_id].set_offsets(np.empty((0, 2)))
                
                # Add player ID labels
                for team_id, positions in team_positions.items():
                    color = team_colors.get(team_id, (0.5, 0.5, 0.5))  # Default gray
                    player_ids = team_player_ids[team_id]
                    
                    for i, (x, y) in enumerate(positions):
                        player_id = player_ids[i]
                        label = ax.text(x, y+150, str(player_id), color='white', fontsize=9, 
                                       ha='center', va='center', bbox=dict(facecolor=color, alpha=0.6))
                        player_labels[f"{team_id}_{player_id}"] = label
            
            # Update ball position
            if frame_idx in ball_positions:
                position = ball_positions[frame_idx]
                x = float(position[0][0])
                y = float(position[0][1])
                
                # Apply vertical flip if requested
                if vertical_flip:
                    y = pitch_width - y
                
                ball_scatter.set_offsets([[x, y]])
                
                # Get recent positions for the ball trail
                trail_frames = [f for f in sorted(ball_positions.keys()) 
                               if f <= frame_idx and f > frame_idx - trail_length]
                trail_x = []
                trail_y = []
                
                for idx in trail_frames:
                    pos = ball_positions[idx]
                    tx = float(pos[0][0])
                    ty = float(pos[0][1])
                    if vertical_flip:
                        ty = pitch_width - ty
                    trail_x.append(tx)
                    trail_y.append(ty)
                
                # Update the trail
                ball_trail.set_data(trail_x, trail_y)
                
                # Store valid trail for future frames
                last_valid_trail_x = trail_x
                last_valid_trail_y = trail_y
            else:
                # If ball is not in this frame, make it invisible
                ball_scatter.set_offsets(np.empty((0, 2)))
                
                # Keep the last valid trail
                ball_trail.set_data(last_valid_trail_x, last_valid_trail_y)
            
            return [title, ball_scatter, ball_trail] + list(scatter_objects.values()) + list(player_labels.values())
        
        # Use ALL player frames, not just common frames
        frames_to_use = list(range(total_player_frames))
        print(f"Using all {len(frames_to_use)} player frames, {len(ball_positions)} have ball positions")
        
        # Determine if we need to reduce frames for performance
        max_frames = 750  # Set to match your total frame count if possible
        if len(frames_to_use) > max_frames:
            frame_step = len(frames_to_use) // max_frames
            frames_to_use = frames_to_use[::frame_step]
            print(f"Using {len(frames_to_use)} frames out of {total_player_frames} (every {frame_step}th frame)")
        
        # Create the animation
        print(f"Creating animation with {len(frames_to_use)} frames at {fps} fps...")
        
        # Set interval based on fps (1000ms / fps)
        interval = 1000 / fps
        
        anim = animation.FuncAnimation(fig, update, frames=frames_to_use, 
                                      init_func=init, blit=True, interval=interval)
        
        # Save the animation
        if output_path:
            print(f"Saving animation to {output_path}...")
            
            try:
                # Configure FFMpeg writer with exact framerate
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Football Analyzer'), bitrate=2000)
                anim.save(output_path, writer=writer, dpi=150)
                print(f"Animation saved to {output_path} at {fps} fps")
            except Exception as e:
                print(f"Error with ffmpeg writer: {e}")
                try:
                    # Fall back to pillow writer
                    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
                except Exception as e:
                    print(f"Error with pillow writer: {e}")
                    # Last resort: save as GIF
                    gif_path = os.path.splitext(output_path)[0] + '.gif'
                    print(f"Trying to save as GIF: {gif_path}")
                    anim.save(gif_path, writer='pillow', fps=5)
            
            print(f"Animation saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

# Run the combined visualization
if __name__ == "__main__":
    # Paths to data
    players_data_path = "/home/elmehdi/Desktop/footballe_analyseur/updated_homographed_players.pkl"
    ball_data_path = "/home/elmehdi/Desktop/footballe_analyseur/homography/homography.pkl"
    
    # Output settings
    output_dir = "/home/elmehdi/Desktop/footballe_analyseur/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "players_and_ball_visualization.mp4")
    
    # Define players to exclude
    excluded_players = ["21", "20", "3", "105", "127", "121", "297", "253", "353","286","278","116","19","262","300","329","333","376"]
    
    # Create the combined visualization
    visualize_players_and_ball(
        players_data_path, 
        ball_data_path, 
        output_path, 
        vertical_flip=True, 
        excluded_players=excluded_players, 
        fps=25
    )