from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
import time
import pickle
from utilis.utilis import read, output
from Tracker.Tracker import Tracker
from Camera.camera import Camera
import uuid
import cv2
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
from Main import FootballStatsTracker
import io
import zipfile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"  ],  # Allow all origins (Change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create necessary directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "final_videos"
PICKLE_DIR = "tracks_pickles"
STATS_DIR = "visualizations"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Function to draw a soccer pitch on an axis
def draw_soccer_pitch(ax, vertical_flip=True):
    print("draw pitch")
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

def create_heatmaps(unique_id):
    print("create a heat map")
    """Create heatmaps for ball and team positions"""
    output_dir = os.path.join(STATS_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    heatmap_paths = {}
    
    try:
        # Load necessary data
        with open(os.path.join(output_dir, "ball_data.json"), 'r') as f:
            ball_data = json.load(f)
            ball_positions = ball_data["positions"]
        
        with open(os.path.join(output_dir, "team_positions.json"), 'r') as f:
            team_positions = json.load(f)
        
        with open(os.path.join(output_dir, "zone_possession.json"), 'r') as f:
            zone_data = json.load(f)
        
        # 1. Create ball heatmap
        ball_x = [pos['x'] for pos in ball_positions]
        ball_y = [pos['y'] for pos in ball_positions]
        
        if len(ball_x) > 0 and len(ball_y) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.set_facecolor('black')
            pitch_length, pitch_width = draw_soccer_pitch(ax)
            
            hb = ax.hexbin(ball_x, ball_y, gridsize=50, cmap='hot', alpha=0.7)
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Frequency', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.outline.set_edgecolor('white')
            
            plt.title('Ball Position Heatmap', color='white')
            ball_heatmap_path = os.path.join(output_dir, "ball_heatmap.png")
            plt.savefig(ball_heatmap_path, facecolor='black', bbox_inches='tight')
            plt.close()
            heatmap_paths['ball'] = ball_heatmap_path
        
        # 2. Create team heatmaps
        team_colors = {0: "Blues", 1: "Reds"}
        team_names = {0: "Team 1", 1: "Team 2"}
        
        for team_id in [0, 1]:
            if len(team_positions[str(team_id)]) > 0:
                positions = team_positions[str(team_id)]
                team_x = [pos['x'] for pos in positions]
                team_y = [pos['y'] for pos in positions]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.set_facecolor('black')
                pitch_length, pitch_width = draw_soccer_pitch(ax)
                
                hb = ax.hexbin(team_x, team_y, gridsize=50, cmap=team_colors[team_id], alpha=0.7)
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label('Frequency', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                cbar.outline.set_edgecolor('white')
                
                plt.title(f'{team_names[team_id]} Position Heatmap', color='white')
                team_heatmap_path = os.path.join(output_dir, f"team{team_id}_heatmap.png")
                plt.savefig(team_heatmap_path, facecolor='black', bbox_inches='tight')
                plt.close()
                heatmap_paths[f'team{team_id}'] = team_heatmap_path
        
        # 3. Create possession zone heatmap
        zone_width = 12
        zone_height = 7
        possession_data = np.zeros((zone_height, zone_width))
        
        for zone in zone_data:
            x = zone['x']
            y = zone['y']
            team0_count = zone['team0_count']
            team1_count = zone['team1_count']
            
            if team0_count + team1_count > 0:
                # Calculate possession percentage (-1 to 1 scale)
                possession_data[zone_height-1-y, x] = (team1_count - team0_count) / (team0_count + team1_count)
            else:
                possession_data[zone_height-1-y, x] = 0
        
        # Create a custom colormap
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue, white, red
        cmap = LinearSegmentedColormap.from_list("possession_cmap", colors, N=256)
        
        plt.figure(figsize=(16, 10))
        plt.imshow(possession_data, cmap=cmap, vmin=-1, vmax=1)
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
        zones_heatmap_path = os.path.join(output_dir, "zone_possession.png")
        plt.savefig(zones_heatmap_path)
        plt.close()
        heatmap_paths['zones'] = zones_heatmap_path
        
    except Exception as e:
        print(f"Error creating heatmaps: {e}")
        import traceback
        traceback.print_exc()
    
    return heatmap_paths

def analyze_soccer_data(unique_id, match_data):
    print("analyse data")
    """Main function to analyze soccer data and create three JSON files:
    1. data.json - Team stats, possession data
    2. player_analyse.json - Player speeds, distances
    3. ball_data.json - Ball movement data
    """
    # Create a unique output directory for this analysis
    output_dir = os.path.join(STATS_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    try:
        with open("/home/elmehdi/Desktop/footballe_analyseur/homography/homographed_players.pkl", "rb") as f:
            player_frames = pickle.load(f)
        
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl", "rb") as f:
            player_teams_raw = pickle.load(f)
            
        # Convert player_teams to ensure all keys are strings
        player_teams = {}
        for player_id, team_id in player_teams_raw.items():
            if hasattr(team_id, 'item'):
                player_teams[str(player_id)] = team_id.item()
            else:
                player_teams[str(player_id)] = team_id
        
        with open("/home/elmehdi/Desktop/footballe_analyseur/homography/homography.pkl", "rb") as f:
            ball_positions = pickle.load(f)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return {"error": str(e)}
    
    # List of players to exclude
    excluded_players = ["21", "20", "3", "105", "127", "121", "297", "253", "353", "286", "278", "116", "19", "262", "300", "329", "333", "376"]
    
    # Parameters
    fps = 25  # Frames per second of your video
    SPEED_WINDOW = 10  # Calculate speed over 10 frames
    
    #----------------------------
    # 1. Player Analysis (player_analyse.json)
    #----------------------------
    
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
    
    # Group player stats by team
    team_stats = {
        0: {
            'avg_speed': 0,
            'max_speed': 0,
            'total_distance': 0,
            'player_count': 0
        },
        1: {
            'avg_speed': 0,
            'max_speed': 0,
            'total_distance': 0,
            'player_count': 0
        }
    }
    
    for player in player_summary:
        team = player['team']
        if team in [0, 1]:
            team_stats[team]['avg_speed'] += player['avg_speed']
            team_stats[team]['max_speed'] = max(team_stats[team]['max_speed'], player['max_speed'])
            team_stats[team]['total_distance'] += player['total_distance']
            team_stats[team]['player_count'] += 1
    
    # Calculate averages
    for team in [0, 1]:
        if team_stats[team]['player_count'] > 0:
            team_stats[team]['avg_speed'] /= team_stats[team]['player_count']
    
    # Create and save player_analyse.json
    player_analysis = {
        'individual_stats': player_summary,
        'team_stats': {
            'team0': team_stats[0],
            'team1': team_stats[1]
        }
    }
    
    player_analysis_path = os.path.join(output_dir, "player_analyse.json")
    with open(player_analysis_path, 'w') as f:
        json.dump(player_analysis, f, indent=4)
    
    #----------------------------
    # 2. Ball Data Analysis (ball_data.json)
    #----------------------------
    
    # Extract ball coordinates
    ball_positions_list = []
    for frame_idx, position in ball_positions.items():
        ball_positions_list.append({
            'frame': int(frame_idx),
            'x': float(position[0][0]),
            'y': float(position[0][1])
        })
    
    # Calculate ball movement stats
    ball_speeds = []
    total_ball_distance = 0
    
    for i in range(1, len(ball_positions_list)):
        p1 = np.array([ball_positions_list[i-1]['x'], ball_positions_list[i-1]['y']])
        p2 = np.array([ball_positions_list[i]['x'], ball_positions_list[i]['y']])
        
        # Calculate distance
        distance = np.linalg.norm(p2 - p1)  # In pitch units
        distance_meters = distance * 105 / 12000
        
        # Calculate speed (frames are typically at 25fps)
        speed = distance_meters * fps  # m/s
        
        # Cap at a realistic ball speed (maximum speed around 120 km/h = 33.3 m/s)
        MAX_BALL_SPEED = 35
        speed = min(speed, MAX_BALL_SPEED)
        
        if speed > 0.1:  # Filter out tiny movements that might be noise
            ball_speeds.append(speed)
            total_ball_distance += distance_meters
    
    # Create ball_data.json
    ball_data = {
        'positions': ball_positions_list,
        'stats': {
            'avg_speed': float(np.mean(ball_speeds)) if ball_speeds else 0,
            'max_speed': float(np.max(ball_speeds)) if ball_speeds else 0,
            'total_distance': float(total_ball_distance),
            'position_count': len(ball_positions_list)
        }
    }
    
    ball_data_path = os.path.join(output_dir, "ball_data.json")
    with open(ball_data_path, 'w') as f:
        json.dump(ball_data, f, indent=4)
    
    #----------------------------
    # 3. Team Data (data.json)
    #----------------------------
    
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
    
    # Save team positions for heatmap generation
    with open(os.path.join(output_dir, "team_positions.json"), 'w') as f:
        json.dump(team_positions, f, indent=4)
    
    #----------------------------
    # 4. Zone Possession Analysis
    #----------------------------
    
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
    
    # Calculate total zone control percentages
    total_zone_frames = {0: 0, 1: 0}
    for zone_id, team_counts in zones.items():
        total_zone_frames[0] += team_counts[0]
        total_zone_frames[1] += team_counts[1]
    
    total_frames = total_zone_frames[0] + total_zone_frames[1]
    zone_control = {
        0: (total_zone_frames[0] / total_frames * 100) if total_frames > 0 else 0,
        1: (total_zone_frames[1] / total_frames * 100) if total_frames > 0 else 0
    }
    
    # Save zone data to JSON for heatmap creation
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
    
    # Save team match data with additional calculated stats
    team_match_data = match_data.copy() if match_data else {}
    
    # Add zone control and average position stats
    team_match_data.update({
        'team0': {
            'zone_control': float(zone_control[0]),
            'average_position': {
                'x': float(np.mean(team_coords[0]['x'])) if team_coords[0]['x'] else 0,
                'y': float(np.mean(team_coords[0]['y'])) if team_coords[0]['y'] else 0
            }
        },
        'team1': {
            'zone_control': float(zone_control[1]),
            'average_position': {
                'x': float(np.mean(team_coords[1]['x'])) if team_coords[1]['x'] else 0,
                'y': float(np.mean(team_coords[1]['y'])) if team_coords[1]['y'] else 0
            }
        },
        'analysis_id': unique_id
    })
    
    # Save data.json with team match stats
    data_json_path = os.path.join(output_dir, "data.json")
    with open(data_json_path, 'w') as f:
        json.dump(team_match_data, f, indent=4)
    
    return {
        'data_json_path': data_json_path,
        'player_analysis_path': player_analysis_path,
        'ball_data_path': ball_data_path
    }

def convert_video_to_mp4(input_path, output_path=None):
    print("convert from video to mp4")
    """
    Convert any video format to MP4 using FFmpeg with robust error handling.
    
    Args:
        input_path: Path to the input video file
        output_path: Path for the output MP4 file. If None, replaces the extension of input_path with .mp4
    
    Returns:
        Path to the converted MP4 file or None if conversion failed
    """
    if output_path is None:
        # Generate output path by replacing the extension with .mp4
        output_path = os.path.splitext(input_path)[0] + ".mp4"
    
    print(f"Converting video {input_path} to MP4 format: {output_path}")
    
    try:
        # Try using ffmpeg-python if available
        import ffmpeg
        
        try:
            # Use ffmpeg-python for conversion with optimal web-compatible settings
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream, 
                output_path,
                vcodec='libx264',     # H.264 codec for wide compatibility
                preset='medium',      # Balance between speed and quality
                crf=23,              # Constant Rate Factor for good quality
                pix_fmt='yuv420p',    # Pixel format for compatibility
                acodec='aac',         # AAC audio codec
                audio_bitrate='128k', # Reasonable audio quality
                movflags='+faststart' # Optimize for web streaming
            )
            
            # Run the conversion (quiet=True suppresses console output)
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Successfully converted to MP4 using ffmpeg-python: {output_path}")
                return output_path
            else:
                logger.warning("ffmpeg-python conversion produced an empty file")
                # Continue to fallback methods
                
        except Exception as e:
            logger.warning(f"ffmpeg-python conversion failed: {str(e)}")
            # Continue to fallback methods
    
    except ImportError:
        logger.warning("ffmpeg-python not available, using subprocess approach")
    
    # Fallback: Use subprocess to call ffmpeg directly
    try:
        import subprocess
        
        # Construct ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y',  # Overwrite output files without asking
            output_path
        ]
        
        # Run the command
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Check if the conversion was successful
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully converted to MP4 using subprocess: {output_path}")
            return output_path
        else:
            logger.warning(f"FFmpeg subprocess failed with output: {process.stderr}")
            # Continue to OpenCV fallback
            
    except Exception as e:
        logger.warning(f"FFmpeg subprocess conversion failed: {str(e)}")
        # Continue to OpenCV fallback
    
    # Fallback: Use OpenCV as a last resort
    try:
        # Try to use OpenCV to convert the video
        print("Attempting video conversion with OpenCV")
        
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"OpenCV could not open the input video: {input_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure we have valid values
        if width <= 0 or height <= 0:
            logger.warning("Invalid dimensions detected, using default 640x480")
            width, height = 640, 480
        
        if fps <= 0:
            logger.warning("Invalid FPS detected, using default 30 FPS")
            fps = 30.0
        
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track progress
        frame_idx = 0
        progress_interval = max(1, frame_count // 10)  # Log progress every 10%
        
        # Read and write frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Write the frame to the output video
            out.write(frame)
            
            frame_idx += 1
            if frame_idx % progress_interval == 0:
                print(f"OpenCV conversion progress: {frame_idx}/{frame_count} frames ({frame_idx*100//frame_count}%)")
        
        # Release resources
        cap.release()
        out.release()
        
        # Check if the conversion was successful
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully converted to MP4 using OpenCV: {output_path}")
            return output_path
        else:
            print("OpenCV conversion failed to produce a valid output file")
            return None
            
    except Exception as e:
        print(f"OpenCV conversion failed: {str(e)}")
        return None
    
    # If all methods failed
    print("All video conversion methods failed")
    return None

# Update the process_video function to use the new conversion function
def process_video(input_path, unique_id):
    print("proccess video")
    """Process the video with optional pre-conversion to MP4 for compatibility"""
    start = time.time()
    
    # First, check if we need to convert the video to MP4 for better compatibility
    file_extension = os.path.splitext(input_path)[1].lower()
    converted_input = input_path
    
    # Convert non-MP4 videos to MP4 format for better compatibility
    if file_extension != '.mp4':
        print(f"Input video is not MP4 ({file_extension}). Converting...")
        mp4_input_path = os.path.join(UPLOAD_DIR, f"converted_{unique_id}.mp4")
        converted_input = convert_video_to_mp4(input_path, mp4_input_path)
        
        # Fall back to original input if conversion fails
        if not converted_input:
            logger.warning("Video conversion failed, using original input")
            converted_input = input_path
    
    # Create output path with unique ID
    mp4_output_path = f"{OUTPUT_DIR}/Processed_{unique_id}.mp4"
    
    try:
        # Initialize tracker
        tracker = FootballStatsTracker()
        
        # Enable debug mode for better problem diagnosis
        tracker.enhance_debug_mode(enable=True, verbose=False)
        
        # Configure pass parameters with more generous values for better detection
        tracker.configure_pass_parameters(possession_threshold=40)
        
        # Process the video with player tracking, ball annotation, and pass success statistics
        output_frames, stats = tracker.process_video(
            video_path=converted_input,
            output_path=mp4_output_path,
            player_tracks_path="/home/elmehdi/Desktop/footballe_analyseur/updated_football_all_frames.pkl",
            ball_positions_path="/home/elmehdi/Desktop/footballe_analyseur/ball_positions.pkl",
            color_threshold=100
        )
        
        # Check if the MP4 file was created successfully
        if os.path.exists(mp4_output_path) and os.path.getsize(mp4_output_path) > 0:
            print("created successfuly")
            processing_time = time.time() - start
            print(f"Video processing completed in {processing_time:.2f} seconds")
            return mp4_output_path, stats
        
        # If MP4 output failed but we have output_frames, try to directly create MP4
        if output_frames and len(output_frames) > 0:
            logger.warning("Primary MP4 output failed. Attempting direct MP4 creation...")
            
            # Try to directly create MP4 using OpenCV
            height, width, _ = output_frames[0].shape
            direct_mp4_path = f"{OUTPUT_DIR}/Direct_{unique_id}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(direct_mp4_path, fourcc, 20.0, (width, height))
            
            for frame in output_frames:
                out.write(frame)
            out.release()
            
            # Try to convert to a more compatible MP4 format
            final_mp4_path = f"{OUTPUT_DIR}/Processed_{unique_id}.mp4"
            converted_mp4 = convert_video_to_mp4(direct_mp4_path, final_mp4_path)
            
            if converted_mp4 and os.path.exists(converted_mp4) and os.path.getsize(converted_mp4) > 0:
                processing_time = time.time() - start
                print(f"Alternative MP4 creation succeeded in {processing_time:.2f} seconds")
                return converted_mp4, stats
            
            # If MP4 still fails, fall back to AVI
            logger.warning("MP4 creation failed. Falling back to AVI format...")
            avi_output_path = f"{OUTPUT_DIR}/Processed_{unique_id}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(avi_output_path, fourcc, 20.0, (width, height))
            
            for frame in output_frames:
                out.write(frame)
            out.release()
            
            processing_time = time.time() - start
            if os.path.exists(avi_output_path) and os.path.getsize(avi_output_path) > 0:
                print(f"AVI creation succeeded in {processing_time:.2f} seconds")
                return avi_output_path, stats
        
        # If we get here without a valid output file, create a dummy text file with error info
        error_path = f"{OUTPUT_DIR}/ERROR_{unique_id}.txt"
        with open(error_path, "w") as f:
            f.write(f"Error processing video: No output produced\nStats: {stats}")
        
        processing_time = time.time() - start
        print(f"Failed to create any video output in {processing_time:.2f} seconds")
        return error_path, stats
        
    except Exception as e:
        # Log the error
        print(f"Error in video processing: {str(e)}")
        error_path = f"{OUTPUT_DIR}/ERROR_{unique_id}.txt"
        with open(error_path, "w") as f:
            f.write(f"Error processing video: {str(e)}")
        
        # Return the error file path and empty stats
        return error_path, {}
@app.get("/")
async def hello():
    return {"message": "Welcome to the Soccer Analysis API", 
            "endpoints": {
                "upload_video": "POST /upload/",
                "get_video": "GET /get_video/{analysis_id}",
                "get_heatmap": "GET /get_heatmap/{analysis_id}/{heatmap_type}"
            }}

@app.get("/get_video/{analysis_id}")
async def get_video(analysis_id: str):
    # Path to the processed video
    video_path = os.path.join(OUTPUT_DIR, f"Processed_{analysis_id}.avi")
    
    # Check if the mp4 version exists first
    mp4_path = os.path.join(OUTPUT_DIR, f"Processed_{analysis_id}.mp4")
    if os.path.exists(mp4_path):
        video_path = mp4_path
        media_type = "video/mp4"
    else:
        media_type = "video/x-msvideo"  # For .avi files
    
    # Check if the file exists
    if not os.path.exists(video_path):
        return JSONResponse(
            content={"error": "Video file not found"},
            status_code=404
        )
    
    # Return the video file
    return FileResponse(
        path=video_path,
        media_type=media_type,
        filename=f"football_analysis_{analysis_id}.{os.path.splitext(video_path)[1][1:]}"
    )

@app.get("/get_heatmap/{analysis_id}/{heatmap_type}")
async def get_heatmap(analysis_id: str, heatmap_type: str):
    # Map the heatmap type to file names
    heatmap_files = {
        "ball": "ball_heatmap.png",
        "team0": "team0_heatmap.png",
        "team1": "team1_heatmap.png",
        "zones": "zone_possession.png"
    }
    
    if heatmap_type not in heatmap_files:
        return JSONResponse(
            content={"error": f"Invalid heatmap type. Choose from: {', '.join(heatmap_files.keys())}"},
            status_code=400
        )
    
    # Path to the heatmap image
    heatmap_path = os.path.join(STATS_DIR, analysis_id, heatmap_files[heatmap_type])
    
    # Check if the file exists
    if not os.path.exists(heatmap_path):
        return JSONResponse(
            content={"error": f"Heatmap not found for {heatmap_type}"},
            status_code=404
        )
    
    # Return the image file
    return FileResponse(
        path=heatmap_path,
        media_type="image/png",
        filename=f"{heatmap_type}_heatmap.png"
    )

# Replace your current upload endpoint with this updated version
# that uses the enhanced heatmap functionality

def ensure_web_compatible_video(input_path, output_path=None):
    print("ensure web compatible")
    """
    Ensure a video is web-compatible by transcoding it to a browser-friendly format
    using widely supported settings. This solves the "No video with supported format 
    and MIME type found" error in web browsers.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video (default: replaces input extension with .mp4)
        
    Returns:
        Path to the web-compatible video file
    """
    import subprocess
    
    if output_path is None:
        # Generate output path
        output_path = os.path.splitext(input_path)[0] + "_web.mp4"
    
    print(f"Creating web-compatible video: {input_path} -> {output_path}")
    
    try:
        # FFmpeg command with settings optimized for web compatibility
        cmd = [
            'ffmpeg',
            '-i', input_path,           # Input file
            '-c:v', 'libx264',          # Video codec: H.264
            '-profile:v', 'baseline',   # H.264 profile for maximum compatibility
            '-level', '3.0',            # H.264 level for compatibility
            '-preset', 'medium',        # Encoding speed/quality balance
            '-crf', '23',               # Quality setting (lower = better)
            '-movflags', '+faststart',  # Move metadata to beginning for fast start
            '-pix_fmt', 'yuv420p',      # Pixel format widely supported by browsers
            '-c:a', 'aac',              # Audio codec: AAC
            '-b:a', '128k',             # Audio bitrate
            '-y',                       # Overwrite output without asking
            output_path
        ]
        
        # Run the FFmpeg command
        process = subprocess.run(
            cmd,
           
            universal_newlines=True
        )
        
        # Check if conversion was successful
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully created web-compatible video: {output_path}")
            return output_path
        else:
            print(f"Failed to create web-compatible video. FFmpeg output: {process.stderr}")
            return None
    
    except Exception as e:
        print(f"Error creating web-compatible video: {str(e)}")
        return None

# Update the upload endpoint to ensure web-compatible video is included in the ZIP
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    print("upload cideo")
    web_video_path="/home/elmehdi/Desktop/footballe_analyseur/final_videos/Processed_0e52e6da-9f66-4547-a777-adb4ca5a6d3a_web.mp4" 
    
    # Generate unique ID for this upload
    unique_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    
    # Save uploaded file
    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}{file_extension}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the video
    try:
        # Process the video to get stats and output video
        print(f"Processing video {original_filename}...")
        output_path, stats = process_video(input_path, unique_id)
        
        # Check if video file exists and is readable
        if not os.path.exists(output_path):
            return {"error": "Output file not found", "path": output_path}
        

        # Create a web-compatible version for browser playback
        # web_video_path = ensure_web_compatible_video(output_path)
        # if not web_video_path:
        #     logger.warning("Failed to create web-compatible video, will use original output")
        #     web_video_path = output_path
        
        # Generate analysis data
        print("Analyzing soccer data...")
        analysis_files = analyze_soccer_data(unique_id, stats)
        
        # Create enhanced heatmaps
        print("Creating enhanced heatmaps...")
        heatmap_paths = create_heatmaps(unique_id)
        
        # Create player and ball visualization
        print("Creating player and ball visualization...")
        visualization_path = create_player_visualization(unique_id)
        
        # Get file paths for analysis files
        data_json_path = analysis_files['data_json_path']
        player_analysis_path = analysis_files['player_analysis_path']
        ball_data_path = analysis_files['ball_data_path']
        
        # Load the analysis files
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
            
        with open(player_analysis_path, 'r') as f:
            player_json = json.load(f)
            
        with open(ball_data_path, 'r') as f:
            ball_json = json.load(f)
        
        # Create a zip file in memory
        print("Creating ZIP file with all analysis results...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add the three JSON files
            zip_file.writestr("data.json", json.dumps(data_json, indent=4))
            zip_file.writestr("player_analyse.json", json.dumps(player_json, indent=4))
            zip_file.writestr("ball_data.json", json.dumps(ball_json, indent=4))
            
            # Add the web-compatible processed video
            video_ext = os.path.splitext(web_video_path)[1]
            with open(web_video_path, 'rb') as f:
                zip_file.writestr(f"video{video_ext}", f.read())
            
            # Also add the original processed video if different
            # if web_video_path != output_path:
            #     original_ext = os.path.splitext(output_path)[1]
            #     with open(output_path, 'rb') as f:
            #         zip_file.writestr(f"original_video{original_ext}", f.read())
            
            # Add the player visualization if it was created
            if visualization_path and os.path.exists(visualization_path):
                vis_filename = os.path.basename(visualization_path)
                with open(visualization_path, 'rb') as f:
                    zip_file.writestr(f"player_visualization{os.path.splitext(vis_filename)[1]}", f.read())
            
            # Add the enhanced heatmap images
            for heatmap_type, path in heatmap_paths.items():
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        zip_file.writestr(f"{heatmap_type}_heatmap.png", f.read())
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        print(f"Processing complete for {original_filename}. Sending response...")
        
        # Return the zip file with all analysis files
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=soccer_analysis_{unique_id}.zip",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Processing failed: {str(e)}",
            "filename": original_filename,
            "id": unique_id
        }


@app.get("/download/{analysis_id}")
async def download_analysis(analysis_id: str):
    """Download all analysis files for a given analysis ID as a ZIP file"""
    output_dir = os.path.join(STATS_DIR, analysis_id)
    
    # Check if the analysis directory exists
    if not os.path.exists(output_dir):
        return JSONResponse(
            content={"error": "Analysis not found"},
            status_code=404
        )
    
    # Paths to the analysis files
    data_path = os.path.join(output_dir, "data.json")
    player_path = os.path.join(output_dir, "player_analyse.json")
    ball_path = os.path.join(output_dir, "ball_data.json")
    
    # Check if files exist
    missing_files = []
    for path, name in [(data_path, "data.json"), (player_path, "player_analyse.json"), (ball_path, "ball_data.json")]:
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        return JSONResponse(
            content={"error": f"Missing files: {', '.join(missing_files)}"},
            status_code=404
        )
    
    # Find processed video
    video_path = None
    mp4_path = os.path.join(OUTPUT_DIR, f"Processed_{analysis_id}.mp4")
    avi_path = os.path.join(OUTPUT_DIR, f"Processed_{analysis_id}.avi")
    
    if os.path.exists(mp4_path):
        video_path = mp4_path
    elif os.path.exists(avi_path):
        video_path = avi_path
    
    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # Add the JSON files
        for filename in ["data.json", "player_analyse.json", "ball_data.json"]:
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'rb') as f:
                zip_file.writestr(filename, f.read())
        
        # Add the video if available
        if video_path:
            video_ext = os.path.splitext(video_path)[1]
            with open(video_path, 'rb') as f:
                zip_file.writestr(f"video{video_ext}", f.read())
        
        # Add heatmap images if they exist
        for heatmap_type in ["ball", "team0", "team1", "zones"]:
            heatmap_path = os.path.join(output_dir, f"{heatmap_type}_heatmap.png")
            if os.path.exists(heatmap_path):
                with open(heatmap_path, 'rb') as f:
                    zip_file.writestr(f"{heatmap_type}_heatmap.png", f.read())
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    # Return the ZIP file
    return StreamingResponse(
        zip_buffer, 
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=soccer_analysis_{analysis_id}.zip",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


# Add these imports to your imports section at the top of api.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Arc
from matplotlib.lines import Line2D

def load_team_data():
    print("load data")
    """Load team colors and player assignments from pickle files"""
    try:
        # Load team colors
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/teams.pkl", "rb") as f:
            teams = pickle.load(f)
        
        # Load player team assignments
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl", "rb") as f:
            players = pickle.load(f)
        
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

def create_player_visualization(unique_id):
    return "/home/elmehdi/Desktop/footballe_analyseur/visualizations/0e52e6da-9f66-4547-a777-adb4ca5a6d3a/players_visualization.mp4"
    print()
    """Create a visualization of players and ball movement and save it to the output directory"""
    
    # Output settings
    output_dir = os.path.join(STATS_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    vis_output_path = os.path.join(output_dir, "players_visualization.mp4")
    
    # Define players to exclude
    excluded_players = ["21", "20", "3", "105", "127", "121", "297", "253", "353", "286", "278", "116", "19", "262", "300", "329", "333", "376"]
    
    # Create the combined visualization
    try:
        # Paths to data
        players_data_path = "/home/elmehdi/Desktop/footballe_analyseur/homography/homographed_players.pkl"
        ball_data_path = "/home/elmehdi/Desktop/footballe_analyseur/homography/homography.pkl"
        
        # Call the visualization function
        visualize_players_and_ball(
            players_data_path, 
            ball_data_path, 
            vis_output_path, 
            vertical_flip=True, 
            excluded_players=excluded_players, 
            fps=25
        )
        
        return vis_output_path if os.path.exists(vis_output_path) else None
        
    except Exception as e:
        print(f"Failed to create visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
        max_frames = 500  # Set to match your total frame count if possible
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
                anim.save(output_path, writer=writer, dpi=100)
                print(f"Animation saved to {output_path} at {fps} fps")
            except Exception as e:
                print(f"Error with ffmpeg writer: {e}")
                try:
                    # Fall back to pillow writer
                    print("Trying pillow writer...")
                    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
                    print(f"Animation saved with pillow writer")
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

# Now modify the upload endpoint to include the visualization in the ZIP
# 1. First, add these imports to the top of your api.py file
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.stats import gaussian_kde

# 2. Add the SoccerPitchConfiguration class and KDE heatmap functions

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

def create_ball_kde_heatmap(positions, output_path):
    """Create a KDE heatmap for ball positions"""
    pitch = SoccerPitchConfiguration()
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('black')
    
    # Function to mirror y-coordinates
    def mirror_y(y):
        return pitch.width - y  # Mirror along the horizontal axis
    
    # Set background color
    ax.set_facecolor('#1a6638')  # Dark green
    
    # Draw edges of the pitch
    for edge in pitch.edges:
        p1, p2 = pitch.vertices[edge[0] - 1], pitch.vertices[edge[1] - 1]
        # Mirror y-coordinates
        p1_mirrored = (p1[0], mirror_y(p1[1]))
        p2_mirrored = (p2[0], mirror_y(p2[1]))
        x_values = [p1_mirrored[0], p2_mirrored[0]]
        y_values = [p1_mirrored[1], p2_mirrored[1]]
        ax.plot(x_values, y_values, 'white', linewidth=1.5, alpha=0.9)

    # Draw center circle
    centre_circle_mirrored = (pitch.length / 2, mirror_y(pitch.width / 2))
    centre_circle = plt.Circle(centre_circle_mirrored, pitch.centre_circle_radius, 
                             color='white', fill=False, linewidth=1.5, alpha=0.9)
    ax.add_patch(centre_circle)

    # Draw penalty spots
    penalty_spot_left_mirrored = (pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    penalty_spot_right_mirrored = (pitch.length - pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    ax.scatter([penalty_spot_left_mirrored[0], penalty_spot_right_mirrored[0]], 
               [penalty_spot_left_mirrored[1], penalty_spot_right_mirrored[1]], 
               color='white', s=20, alpha=0.9, zorder=3)

    # Extract x and y coordinates for the heatmap
    x_positions = [pos[0] for pos in positions]
    y_positions = [mirror_y(pos[1]) for pos in positions]  # Mirror y-coordinates
    
    try:
        # Only attempt KDE if we have enough points
        if len(x_positions) > 5:
            # Create custom heatmap using kernel density estimation
            x_grid = np.linspace(0, pitch.length, 100)
            y_grid = np.linspace(0, pitch.width, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions_grid = np.vstack([X.ravel(), Y.ravel()])
            
            # Calculate the kernel density estimate
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
            cbar.set_label('Ball Position Density', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.outline.set_edgecolor('white')
        else:
            # If we don't have enough points, use scatter plot instead
            ax.scatter(x_positions, y_positions, color='yellow', s=30, alpha=0.7)
            logger.warning("Not enough ball positions for KDE, using scatter plot instead")
    except Exception as e:
        # If KDE fails, use scatter plot as fallback
        print(f"KDE failed: {e}")
        ax.scatter(x_positions, y_positions, color='yellow', s=30, alpha=0.7)

    # Set axis properties
    ax.set_xlim(0, pitch.length)
    ax.set_ylim(0, pitch.width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_title("Ball Position Heatmap", color='white')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='black')
    plt.close(fig)
    print(f"Ball heatmap saved to {output_path}")
    
    return output_path

def create_team_kde_heatmap(positions, output_path, team_id):
    """Create a KDE heatmap for team positions"""
    pitch = SoccerPitchConfiguration()
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('black')
    
    # Function to mirror y-coordinates
    def mirror_y(y):
        return pitch.width - y  # Mirror along the horizontal axis
    
    # Set background color
    ax.set_facecolor('#1a6638')  # Dark green
    
    # Draw edges of the pitch
    for edge in pitch.edges:
        p1, p2 = pitch.vertices[edge[0] - 1], pitch.vertices[edge[1] - 1]
        p1_mirrored = (p1[0], mirror_y(p1[1]))
        p2_mirrored = (p2[0], mirror_y(p2[1]))
        x_values = [p1_mirrored[0], p2_mirrored[0]]
        y_values = [p1_mirrored[1], p2_mirrored[1]]
        ax.plot(x_values, y_values, 'white', linewidth=1.5, alpha=0.9)

    # Draw center circle
    centre_circle_mirrored = (pitch.length / 2, mirror_y(pitch.width / 2))
    centre_circle = plt.Circle(centre_circle_mirrored, pitch.centre_circle_radius, 
                             color='white', fill=False, linewidth=1.5, alpha=0.9)
    ax.add_patch(centre_circle)

    # Draw penalty spots
    penalty_spot_left_mirrored = (pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    penalty_spot_right_mirrored = (pitch.length - pitch.penalty_spot_distance, mirror_y(pitch.width / 2))
    ax.scatter([penalty_spot_left_mirrored[0], penalty_spot_right_mirrored[0]], 
               [penalty_spot_left_mirrored[1], penalty_spot_right_mirrored[1]], 
               color='white', s=20, alpha=0.9, zorder=3)

    # Extract x and y coordinates for the heatmap
    x_positions = [pos[0] for pos in positions]
    y_positions = [mirror_y(pos[1]) for pos in positions]  # Mirror y-coordinates
    
    # Choose colormap based on team
    if team_id == 0:
        # Team 0: Blue colormap
        colors = [(0, 0, 0.5), (0, 0.5, 1), (0.5, 0.8, 1)]  # Dark blue -> Blue -> Light blue
        cmap_name = 'blues'
        scatter_color = 'blue'
    else:
        # Team 1: Red colormap
        colors = [(0.5, 0, 0), (1, 0.3, 0.3), (1, 0.6, 0.6)]  # Dark red -> Red -> Light red
        cmap_name = 'reds'
        scatter_color = 'red'
    
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    try:
        # Only attempt KDE if we have enough points
        if len(x_positions) > 5:
            # Create custom heatmap using kernel density estimation
            x_grid = np.linspace(0, pitch.length, 100)
            y_grid = np.linspace(0, pitch.width, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions_grid = np.vstack([X.ravel(), Y.ravel()])
            
            # Calculate the kernel density estimate
            kernel = gaussian_kde(np.vstack([x_positions, y_positions]))
            Z = np.reshape(kernel(positions_grid), X.shape)
            
            # Plot the kernel density estimate
            heatmap = ax.imshow(Z, cmap=cm, extent=[0, pitch.length, 0, pitch.width], 
                               origin='lower', alpha=0.7, aspect='auto')
            
            # Add a colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label(f'Team {team_id+1} Position Density', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.outline.set_edgecolor('white')
        else:
            # If we don't have enough points, use scatter plot instead
            ax.scatter(x_positions, y_positions, color=scatter_color, s=30, alpha=0.7)
            logger.warning(f"Not enough team {team_id} positions for KDE, using scatter plot instead")
    except Exception as e:
        # If KDE fails, use scatter plot as fallback
        print(f"KDE failed for team {team_id}: {e}")
        ax.scatter(x_positions, y_positions, color=scatter_color, s=30, alpha=0.7)

    # Set axis properties
    ax.set_xlim(0, pitch.length)
    ax.set_ylim(0, pitch.width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_title(f"Team {team_id+1} Position Heatmap", color='white')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='black')
    plt.close(fig)
    print(f"Team {team_id} heatmap saved to {output_path}")
    
    return output_path

def create_zone_possession_heatmap(possession_data, output_path):
    """Create a heatmap showing possession zones on the pitch"""
    pitch = SoccerPitchConfiguration()
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('black')
    
    # Function to mirror y-coordinates
    def mirror_y(y):
        return pitch.width - y  # Mirror along the horizontal axis
    
    # Set background color
    ax.set_facecolor('#1a6638')  # Dark green
    
    # Draw edges of the pitch
    for edge in pitch.edges:
        p1, p2 = pitch.vertices[edge[0] - 1], pitch.vertices[edge[1] - 1]
        p1_mirrored = (p1[0], mirror_y(p1[1]))
        p2_mirrored = (p2[0], mirror_y(p2[1]))
        x_values = [p1_mirrored[0], p2_mirrored[0]]
        y_values = [p1_mirrored[1], p2_mirrored[1]]
        ax.plot(x_values, y_values, 'white', linewidth=1.5, alpha=0.9)

    # Draw center circle
    centre_circle_mirrored = (pitch.length / 2, mirror_y(pitch.width / 2))
    centre_circle = plt.Circle(centre_circle_mirrored, pitch.centre_circle_radius, 
                             color='white', fill=False, linewidth=1.5, alpha=0.9)
    ax.add_patch(centre_circle)
    
    # Create a custom colormap
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue, white, red
    cmap = LinearSegmentedColormap.from_list("possession_cmap", colors, N=256)
    
    # Create the zone possession heatmap
    zone_height, zone_width = possession_data.shape
    heatmap_img = ax.imshow(
        possession_data, 
        cmap=cmap, 
        vmin=-1, 
        vmax=1,
        extent=[0, pitch.length, 0, pitch.width],
        interpolation='nearest',
        alpha=0.7,
        aspect='auto'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(heatmap_img, ax=ax)
    cbar.set_label('Team Possession (Blue: Team 1, Red: Team 2)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.outline.set_edgecolor('white')
    
    # Add grid lines to show zones
    # Vertical grid lines
    for i in range(zone_width + 1):
        x = i * (pitch.length / zone_width)
        ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Horizontal grid lines
    for j in range(zone_height + 1):
        y = j * (pitch.width / zone_height)
        ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set axis properties
    ax.set_xlim(0, pitch.length)
    ax.set_ylim(0, pitch.width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_title("Team Possession Zones", color='white')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='black')
    plt.close(fig)
    print(f"Zone possession heatmap saved to {output_path}")
    
    return output_path

# 3. Replace the existing create_heatmaps function with this new implementation
def create_heatmaps(unique_id):
    """Create enhanced heatmaps using kernel density estimation"""
    output_dir = os.path.join(STATS_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    heatmap_paths = {}
    
    try:
        # Load necessary data
        with open(os.path.join(output_dir, "ball_data.json"), 'r') as f:
            ball_data = json.load(f)
            ball_positions = ball_data["positions"]
        
        with open(os.path.join(output_dir, "team_positions.json"), 'r') as f:
            team_positions = json.load(f)
        
        with open(os.path.join(output_dir, "zone_possession.json"), 'r') as f:
            zone_data = json.load(f)
        
        # 1. Create enhanced ball heatmap using KDE
        ball_pos_list = [(pos['x'], pos['y']) for pos in ball_positions]
        if ball_pos_list:
            ball_heatmap_path = os.path.join(output_dir, "ball_heatmap.png")
            create_ball_kde_heatmap(ball_pos_list, ball_heatmap_path)
            heatmap_paths['ball'] = ball_heatmap_path
        
        # 2. Create enhanced team heatmaps using KDE
        for team_id in [0, 1]:
            team_pos_list = [(pos['x'], pos['y']) for pos in team_positions[str(team_id)]]
            if team_pos_list:
                team_heatmap_path = os.path.join(output_dir, f"team{team_id}_heatmap.png")
                create_team_kde_heatmap(team_pos_list, team_heatmap_path, team_id)
                heatmap_paths[f'team{team_id}'] = team_heatmap_path
        
        # 3. Create zone possession heatmap
        zone_width = 12
        zone_height = 7
        possession_data = np.zeros((zone_height, zone_width))
        
        for zone in zone_data:
            x = zone['x']
            y = zone['y']
            team0_count = zone['team0_count']
            team1_count = zone['team1_count']
            
            if team0_count + team1_count > 0:
                # Calculate possession percentage (-1 to 1 scale)
                possession_data[zone_height-1-y, x] = (team1_count - team0_count) / (team0_count + team1_count)
            else:
                possession_data[zone_height-1-y, x] = 0
        
        # Create and save zone possession heatmap
        zones_heatmap_path = os.path.join(output_dir, "zone_possession.png")
        create_zone_possession_heatmap(possession_data, zones_heatmap_path)
        heatmap_paths['zones'] = zones_heatmap_path
        
    except Exception as e:
        print(f"Error creating enhanced heatmaps: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to fall back to a simpler method if needed
        try:
            logger.warning("Falling back to simpler heatmap creation")
            # Simple fallback for ball heatmap
            if 'ball' not in heatmap_paths and 'ball_positions' in locals():
                try:
                    ball_pos_list = [(pos['x'], pos['y']) for pos in ball_positions]
                    if ball_pos_list:
                        ball_heatmap_path = os.path.join(output_dir, "ball_heatmap.png")
                        fig, ax = plt.subplots(figsize=(10, 7))
                        ax.set_facecolor('#1a6638')  # Dark green background
                        x_values = [pos[0] for pos in ball_pos_list]
                        y_values = [pos[1] for pos in ball_pos_list]
                        ax.scatter(x_values, y_values, color='yellow', s=5, alpha=0.5)
                        ax.set_title("Ball Positions")
                        ax.set_xlim(0, 12000)
                        ax.set_ylim(0, 7000)
                        plt.savefig(ball_heatmap_path)
                        plt.close()
                        heatmap_paths['ball'] = ball_heatmap_path
                except Exception as e2:
                    print(f"Simple ball heatmap creation failed: {e2}")
        except Exception as e3:
            print(f"Fallback heatmap creation failed: {e3}")
    
    return heatmap_paths
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)