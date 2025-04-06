import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Arc

# Create output directory if it doesn't exist
output_dir = "/home/elmehdi/Desktop/footballe_analyseur/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Function to draw a soccer pitch on an axis
def draw_soccer_pitch(ax, vertical_flip=True):
    """Draw a simple soccer pitch on the given axes"""
    pitch_length = 12000
    pitch_width = 7000
    
    ax.set_facecolor('#1a6638')  # Dark green
    ax.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], 'white', lw=2)
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], 'white', lw=2)
    
    center_circle = plt.Circle((pitch_length/2, pitch_width/2), 915, color='white', fill=False, lw=2)
    ax.add_patch(center_circle)
    
    # Draw penalty areas
    ax.plot([0, 2015], [7000/2 - 4100/2, 7000/2 - 4100/2], 'white', lw=2)
    ax.plot([2015, 2015], [7000/2 - 4100/2, 7000/2 + 4100/2], 'white', lw=2)
    ax.plot([0, 2015], [7000/2 + 4100/2, 7000/2 + 4100/2], 'white', lw=2)
    
    ax.plot([pitch_length, pitch_length - 2015], [7000/2 - 4100/2, 7000/2 - 4100/2], 'white', lw=2)
    ax.plot([pitch_length - 2015, pitch_length - 2015], [7000/2 - 4100/2, 7000/2 + 4100/2], 'white', lw=2)
    ax.plot([pitch_length, pitch_length - 2015], [7000/2 + 4100/2, 7000/2 + 4100/2], 'white', lw=2)
    
    # Draw penalty spots
    ax.plot(1100, 7000/2, 'wo', markersize=5)
    ax.plot(pitch_length - 1100, 7000/2, 'wo', markersize=5)
    
    ax.set_xlim(-500, pitch_length + 500)
    ax.set_ylim(-500, pitch_width + 500)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return pitch_length, pitch_width

def analyze_soccer_data():
    """Main function to analyze soccer data and create visualizations"""
    print("Starting soccer data analysis...")
    
    # Load the data
    try:
        print("Loading player data...")
        with open("/home/elmehdi/Desktop/footballe_analyseur/homography/homographed_players.pkl", "rb") as f:
            player_frames = pickle.load(f)
        
        print("Loading player team assignments...")
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl", "rb") as f:
            player_teams_raw = pickle.load(f)
            
        # Convert player_teams to ensure all keys are strings
        player_teams = {}
        for player_id, team_id in player_teams_raw.items():
            if hasattr(team_id, 'item'):
                player_teams[str(player_id)] = team_id.item()
            else:
                player_teams[str(player_id)] = team_id
        
        print("Loading ball position data...")
        with open("/home/elmehdi/Desktop/footballe_analyseur/homography/homography.pkl", "rb") as f:
            ball_positions = pickle.load(f)
            
        print("Data loading complete!")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # List of players to exclude
    excluded_players = ["21", "20", "3", "105", "127", "121", "297", "253", "353", "286", "278", "116", "19", "262", "300", "329", "333", "376"]
    
    # Parameters
    fps = 25  # Frames per second of your video
    SPEED_WINDOW = 10  # Calculate speed over 10 frames (0.4 seconds at 25 fps)
    
    #----------------------------
    # 1. Player Speed and Distance Analysis
    #----------------------------
    print("\n1. Calculating player speeds and distances...")
    
    # Initialize player stats
    player_stats = {}
    
    # First pass: identify players and initialize their stats
    for frame in player_frames:
        for player_id, player_info in frame.items():
            str_player_id = str(player_id)
            if isinstance(player_info, dict) and 'pitch_position' in player_info:
                if player_id != "ball" and str_player_id not in excluded_players:
                    if player_id not in player_stats:
                        player_stats[player_id] = {
                            'positions': [],
                            'frame_indices': [],
                            'speeds': [],
                            'distances': [],
                            'total_distance': 0,
                            'team': player_teams.get(str_player_id, None)
                        }
    
    # Second pass: gather positions for each player across frames
    for frame_idx, frame in enumerate(player_frames):
        for player_id, player_info in frame.items():
            if player_id in player_stats and isinstance(player_info, dict) and 'pitch_position' in player_info:
                position = player_info['pitch_position']
                player_stats[player_id]['positions'].append(position)
                player_stats[player_id]['frame_indices'].append(frame_idx)
    
    # Define maximum realistic speed for a soccer player (m/s)
    # World-class sprinters reach about 10-12 m/s
    MAX_REALISTIC_SPEED = 10  # About 43 km/h
    
    # Third pass: calculate distances frame by frame
    for player_id, stats in player_stats.items():
        positions = stats['positions']
        frame_indices = stats['frame_indices']
        
        # Calculate distances between consecutive positions
        distances = []
        for i in range(1, len(positions)):
            p1 = np.array(positions[i-1])
            p2 = np.array(positions[i])
            
            # Calculate distance between positions
            distance = np.linalg.norm(p2 - p1)  # In pitch units
            
            # Convert distance to meters (assuming pitch is in mm and 105x68m standard size)
            # 12000 units = 105 meters width
            distance_meters = distance * 105 / 12000
            
            # Add to distances list
            distances.append(distance_meters)
            
            # Add to total distance
            stats['total_distance'] += distance_meters
        
        stats['distances'] = distances
    
    # Fourth pass: calculate speeds over windows of SPEED_WINDOW frames
    for player_id, stats in player_stats.items():
        positions = stats['positions']
        frame_indices = stats['frame_indices']
        
        # Need at least SPEED_WINDOW+1 positions to calculate speed over SPEED_WINDOW frames
        if len(positions) < SPEED_WINDOW+1:
            continue
        
        speeds = []
        for i in range(len(positions) - SPEED_WINDOW):
            # Get positions SPEED_WINDOW frames apart
            p1 = np.array(positions[i])
            p2 = np.array(positions[i + SPEED_WINDOW])
            
            # Get time difference
            dt = (frame_indices[i + SPEED_WINDOW] - frame_indices[i]) / fps  # Time in seconds
            
            # Calculate direct distance (not summing frame-by-frame distances)
            distance = np.linalg.norm(p2 - p1)  # In pitch units
            distance_meters = distance * 105 / 12000
            
            # Calculate speed over the window
            if dt > 0:
                speed = distance_meters / dt  # m/s
                
                # Cap at maximum realistic speed
                speed = min(speed, MAX_REALISTIC_SPEED)
            else:
                speed = 0
            
            speeds.append(speed)
        
        stats['speeds'] = speeds
    
    # Convert to summarized DataFrame
    player_summary = []
    for player_id, stats in player_stats.items():
        if len(stats['speeds']) > 0:
            avg_speed = np.mean(stats['speeds'])
            max_speed = np.max(stats['speeds'])
            total_distance = stats['total_distance']
            team = stats['team']
            
            player_summary.append({
                'player_id': str(player_id),
                'team': team,
                'avg_speed': float(avg_speed),
                'max_speed': float(max_speed),
                'total_distance': float(total_distance)
            })
    
    player_df = pd.DataFrame(player_summary)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "player_stats.csv")
    player_df.to_csv(csv_path, index=False)
    print(f"Player stats saved to {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(output_dir, "player_stats.json")
    with open(json_path, 'w') as f:
        json.dump(player_summary, f, indent=4)
    print(f"Player stats saved to {json_path}")
    
    # Create visualizations for player stats
    print("Creating player stats visualizations...")
    
    # Sort by average speed and distance
    player_df_speed = player_df.sort_values(by='avg_speed', ascending=False)
    player_df_distance = player_df.sort_values(by='total_distance', ascending=False)
    
    # Top players by average speed
    plt.figure(figsize=(12, 8))
    sns.barplot(x='player_id', y='avg_speed', hue='team', data=player_df_speed.head(10))
    plt.title('Top 10 Players by Average Speed')
    plt.ylabel('Average Speed (m/s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_players_speed.png"))
    plt.close()
    
    # Top players by total distance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='player_id', y='total_distance', hue='team', data=player_df_distance.head(10))
    plt.title('Top 10 Players by Total Distance Covered')
    plt.ylabel('Total Distance (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_players_distance.png"))
    plt.close()
    
    #----------------------------
    # 2. Ball Position Heatmap
    #----------------------------
    print("\n2. Creating ball position heatmap...")
    
    # Extract ball coordinates
    ball_x = []
    ball_y = []
    
    for frame_idx, position in ball_positions.items():
        x = float(position[0][0])
        y = float(position[0][1])
        ball_x.append(x)
        ball_y.append(y)
    
    # Create ball heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('black')
    pitch_length, pitch_width = draw_soccer_pitch(ax)
    
    if len(ball_x) > 0 and len(ball_y) > 0:
        # Ball heatmap
        hb = ax.hexbin(ball_x, ball_y, gridsize=50, cmap='hot', alpha=0.7)
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Frequency', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.outline.set_edgecolor('white')
    
    plt.title('Ball Position Heatmap', color='white')
    plt.savefig(os.path.join(output_dir, "ball_heatmap.png"), facecolor='black', bbox_inches='tight')
    plt.close()
    
    # Save ball positions to JSON
    ball_positions_list = []
    for frame_idx, position in ball_positions.items():
        ball_positions_list.append({
            'frame': int(frame_idx),
            'x': float(position[0][0]),
            'y': float(position[0][1])
        })
    
    with open(os.path.join(output_dir, "ball_positions.json"), 'w') as f:
        json.dump(ball_positions_list, f, indent=4)
    
    #----------------------------
    # 3. Team-Based Player Heatmaps
    #----------------------------
    print("\n3. Creating team-based player heatmaps...")
    
    team_coords = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}}
    team_positions = {0: [], 1: []}
    
    # Collect positions for each team
    for frame in player_frames:
        for player_id, player_info in frame.items():
            str_player_id = str(player_id)
            if isinstance(player_info, dict) and 'pitch_position' in player_info:
                if player_id != "ball" and str_player_id not in excluded_players:
                    team = player_teams.get(str_player_id, None)
                    if team in [0, 1]:  # Only include players from team 0 or 1
                        x, y = player_info['pitch_position']
                        team_coords[team]['x'].append(x)
                        team_coords[team]['y'].append(y)
                        team_positions[team].append({'x': float(x), 'y': float(y)})
    
    # Save team positions to JSON
    with open(os.path.join(output_dir, "team_positions.json"), 'w') as f:
        json.dump(team_positions, f, indent=4)
    
    # Create heatmaps for each team
    team_names = {0: "Team 1", 1: "Team 2"}
    team_colors = {0: "Blues", 1: "Reds"}
    
    for team_id, coords in team_coords.items():
        if len(coords['x']) > 0 and len(coords['y']) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.set_facecolor('black')
            pitch_length, pitch_width = draw_soccer_pitch(ax)
            
            # Team heatmap
            hb = ax.hexbin(coords['x'], coords['y'], gridsize=50, cmap=team_colors[team_id], alpha=0.7)
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Frequency', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.outline.set_edgecolor('white')
            
            plt.title(f'{team_names[team_id]} Position Heatmap', color='white')
            plt.savefig(os.path.join(output_dir, f"team{team_id}_heatmap.png"), 
                       facecolor='black', bbox_inches='tight')
            plt.close()
    
    #----------------------------
    # 4. Individual Player Heatmaps (Top Players)
    #----------------------------
    print("\n4. Creating individual player heatmaps for top players...")
    
    # Create individual heatmaps for top players (by distance covered)
    top_players = player_df.sort_values(by='total_distance', ascending=False).head(5)['player_id'].tolist()
    
    player_position_data = {}
    
    for player_id in top_players:
        # Collect all positions for this player
        positions_x = []
        positions_y = []
        positions = []
        
        for frame in player_frames:
            if player_id in frame and isinstance(frame[player_id], dict) and 'pitch_position' in frame[player_id]:
                x, y = frame[player_id]['pitch_position']
                positions_x.append(x)
                positions_y.append(y)
                positions.append({'x': float(x), 'y': float(y)})
        
        # Save player positions to the dictionary
        player_position_data[player_id] = positions
        
        if len(positions_x) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.set_facecolor('black')
            pitch_length, pitch_width = draw_soccer_pitch(ax)
            
            # Player heatmap
            hb = ax.hexbin(positions_x, positions_y, gridsize=40, cmap='viridis', alpha=0.7)
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Frequency', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.outline.set_edgecolor('white')
            
            team = player_teams.get(str(player_id), "Unknown")
            plt.title(f'Player {player_id} (Team {team}) Position Heatmap', color='white')
            plt.savefig(os.path.join(output_dir, f"player{player_id}_heatmap.png"), 
                       facecolor='black', bbox_inches='tight')
            plt.close()
    
    # Save top player positions to JSON
    with open(os.path.join(output_dir, "top_player_positions.json"), 'w') as f:
        json.dump(player_position_data, f, indent=4)
    
    #----------------------------
    # 5. Team Possession Zones
    #----------------------------
    print("\n5. Analyzing team possession zones...")
    
    # Create a grid of the pitch (e.g., 12x7 zones)
    zone_width = 12
    zone_height = 7
    zones = {}
    
    for x in range(zone_width):
        for y in range(zone_height):
            zone_id = f"{x}_{y}"
            zones[zone_id] = {0: 0, 1: 0}  # Count for each team
    
    # Process each frame to count team presence in zones
    for frame in player_frames:
        # Track which team has more players in each zone for this frame
        frame_zones = {}
        
        for player_id, player_info in frame.items():
            str_player_id = str(player_id)
            if isinstance(player_info, dict) and 'pitch_position' in player_info:
                if player_id != "ball" and str_player_id not in excluded_players:
                    team = player_teams.get(str_player_id, None)
                    if team in [0, 1]:
                        x, y = player_info['pitch_position']
                        
                        # Calculate which zone this position falls into
                        zone_x = min(int(x / (12000 / zone_width)), zone_width - 1)
                        zone_y = min(int(y / (7000 / zone_height)), zone_height - 1)
                        zone_id = f"{zone_x}_{zone_y}"
                        
                        # Increment the count for this team in this zone
                        if zone_id not in frame_zones:
                            frame_zones[zone_id] = {0: 0, 1: 0}
                        frame_zones[zone_id][team] += 1
        
        # For each zone in this frame, increment the team with more players
        for zone_id, team_counts in frame_zones.items():
            if team_counts[0] > team_counts[1]:
                zones[zone_id][0] += 1
            elif team_counts[1] > team_counts[0]:
                zones[zone_id][1] += 1
            # If equal, don't increment (considered contested)
    
    # Save zone data to JSON
    zone_data_list = []
    for zone_id, team_counts in zones.items():
        x, y = map(int, zone_id.split('_'))
        zone_data_list.append({
            'zone_id': zone_id,
            'x': x,
            'y': y,
            'team0_count': team_counts[0],
            'team1_count': team_counts[1]
        })
    
    with open(os.path.join(output_dir, "zone_possession.json"), 'w') as f:
        json.dump(zone_data_list, f, indent=4)
    
    # Create a possession map
    possession_data = np.zeros((zone_height, zone_width))
    for x in range(zone_width):
        for y in range(zone_height):
            zone_id = f"{x}_{y}"
            team0_count = zones[zone_id][0]
            team1_count = zones[zone_id][1]
            
            if team0_count + team1_count > 0:
                # Calculate possession percentage for team 1 (-1 to 1 scale)
                # -1 means 100% team 0, +1 means 100% team 1
                possession_data[zone_height-1-y, x] = (team1_count - team0_count) / (team0_count + team1_count)
            else:
                possession_data[zone_height-1-y, x] = 0
    
    # Create a custom colormap: blue for team 0, red for team 1, white for neutral
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue, white, red
    cmap = LinearSegmentedColormap.from_list("possession_cmap", colors, N=256)
    
    # Plot the possession map
    plt.figure(figsize=(16, 10))
    plt.imshow(possession_data, cmap=cmap, vmin=-1, vmax=1)
    
    # Add labels and grid
    plt.colorbar(label='Team Possession (Blue: Team 0, Red: Team 1)')
    plt.title('Team Possession Zones')
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    # Add axes labels
    x_labels = [f"{int(x * 12000 / zone_width)}" for x in range(zone_width)]
    y_labels = [f"{int(y * 7000 / zone_height)}" for y in range(zone_height)]
    plt.xticks(range(zone_width), x_labels, rotation=45)
    plt.yticks(range(zone_height), reversed(y_labels))
    plt.xlabel('X Coordinate (pitch units)')
    plt.ylabel('Y Coordinate (pitch units)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "team_possession_zones.png"))
    plt.close()
    
    print("\nAnalysis complete! All data and visualizations saved to:", output_dir)
    
    # Return a summary of what was created
    return {
        "player_stats": len(player_summary),
        "ball_positions": len(ball_positions),
        "team_positions": {0: len(team_positions[0]), 1: len(team_positions[1])},
        "zone_data": len(zone_data_list),
        "output_directory": output_dir
    }

# Run the analysis
if __name__ == "__main__":
    summary = analyze_soccer_data()
    print("\nSummary of extracted data:")
    for key, value in summary.items():
        print(f"- {key}: {value}")