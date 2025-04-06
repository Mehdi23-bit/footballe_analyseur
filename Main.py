from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial.distance import cdist
from collections import defaultdict

class Teams:
    def __init__(self):
        self.teams = {}
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl","rb")as f:
            self.players=pickle.load(f)
        
        self.kmean = None
        self.color_threshold = 100  # Threshold for color difference to consider a player an outlier (like goalkeeper)

    def resize(self, image):
        img = image
        height, width, _ = img.shape
        uperhalf = img[:int(height/2), :]
        bigger_image = cv2.resize(uperhalf, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
        return bigger_image

    def get_player_color(self, frame):
        image = frame
        try:
            img = self.resize(image)
            pixels = img.reshape((-1, 3))
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(pixels)
            labels = kmeans.labels_
            clustered_image = labels.reshape(img.shape[0], img.shape[1])
            corner = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
            nplayer = max(set(corner), key=corner.count)
            player_cluster = 1 - nplayer
            return kmeans.cluster_centers_[player_cluster]
        except Exception as e:
            print(f"Error in get_player_color: {e}")
            return np.array([0, 0, 0])  # Default to black in case of error

    def get_teams_color(self, tracks, frame):
        if self.teams:
            return self.teams
        
        colors = []
        # Get the first frame key
        try:
            first_frame = list(tracks.keys())[0]
            
            for player_id, player_data in tracks[first_frame]["players"].items():
                try:
                    # Check which bbox key is available
                    if "original_bbox" in player_data:
                        bbox = player_data["original_bbox"]
                    elif "bbox" in player_data:
                        bbox = player_data["bbox"]
                    else:
                        continue
                        
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Check if the bbox is valid
                    if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or x1 >= x2 or y1 >= y2:
                        continue
                    
                    image = frame[y1:y2, x1:x2]
                    if image.size == 0:
                        continue
                    
                    colors.append(self.get_player_color(image))
                except Exception as e:
                    print(f"Error processing player {player_id}: {e}")
                    continue
        except Exception as e:
            print(f"Error accessing tracks data: {e}")
        
        if len(colors) < 2:
            self.teams[0] = (0, 0, 255)  # Blue
            self.teams[1] = (255, 0, 0)  # Red
            print("Not enough valid players to determine team colors. Using default colors.")
        else:
            kmean = KMeans(n_clusters=2, random_state=0)
            kmean.fit(colors)
            self.kmean = kmean
            self.teams[0] = tuple(map(int, kmean.cluster_centers_[0]))
            self.teams[1] = tuple(map(int, kmean.cluster_centers_[1]))
        with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/teams.pkl","wb")as f:
            pickle.dump(self.teams,f)    
        return self.teams

    def set_players(self, tracks, frame):
        try:
            # Get the first frame key
            first_frame = list(tracks.keys())[0]
            
            for player_id, player_data in tracks[first_frame]["players"].items():
                try:
                    # Check which bbox key is available
                    if "original_bbox" in player_data:
                        bbox = player_data["original_bbox"]
                    elif "bbox" in player_data:
                        bbox = player_data["bbox"]
                    else:
                        continue
                        
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Check if the bbox is valid
                    if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or x1 >= x2 or y1 >= y2:
                        continue
                    
                    image = frame[y1:y2, x1:x2]
                    if image.size == 0:
                        continue
                    
                    color = self.get_player_color(image)
                    
                    # Check if color is too far from team colors (potential goalkeeper)
                    min_distance = self.is_outlier_color(color)
                    if min_distance > self.color_threshold:
                        self.players[str(player_id)] = -1  # Mark as goalkeeper/outlier
                        continue
                    
                    team_id = self.kmean.predict(color.reshape(1, -1))[0]
                    self.players[str(player_id)] = team_id
                except Exception as e:
                    print(f"Error setting team for player {player_id}: {e}")
                    self.players[str(player_id)] = 0  # Default to team 0
        except Exception as e:
            print(f"Error in set_players: {e}")

    def is_outlier_color(self, color):
        """Check if color is significantly different from team colors (goalkeeper)"""
        if not self.kmean or not self.teams:
            return 0
            
        # Calculate distances from both team colors
        distances = []
        for team_id in [0, 1]:
            team_color = np.array(self.teams[team_id])
            distance = np.linalg.norm(color - team_color)
            distances.append(distance)
            
        # Return minimum distance from any team color
        return min(distances)

    def get_player_team(self, frame, player_id, player_bbox):
        player_id_str = str(player_id)
        # print(f"player id is : {player_id_str} ")
        # if player_id_str in self.players and player_id_str not in {"17","10","118","177","8","14","189","140"}:
        
        return self.players[player_id_str]
        
        # # try:
        # #     print("analyse color of player")
        # #     bbox = player_bbox
        # #     x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
        # #     # Check if the bbox is valid
        # #     if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or x1 >= x2 or y1 >= y2:
        # #         return 0  # Default to team 0
            
        # #     image = frame[y1:y2, x1:x2]
        # #     if image.size == 0:
        # #         return 0  # Default to team 0
            
        # #     color = self.get_player_color(image)
        # #     if self.kmean is None:
        # #         return 0  # Default to team 0
            
        # #     # Check if color is too different from team colors (potential goalkeeper)
        # #     min_distance = self.is_outlier_color(color)
        # #     if min_distance > self.color_threshold:
        # #         self.players[player_id_str] = -1  # Mark as goalkeeper/outlier
        # #         return -1
                
        #     team_id = self.kmean.predict(color.reshape(1, -1))[0]
        #     self.players[player_id_str] = team_id
        #     with open("/home/elmehdi/Desktop/footballe_analyseur/tracks_pickles/players_team.pkl","wb")as f:
        #         pickle.dump(self.players,f)
        #     return team_id
        # except Exception as e:
        #     print(f"Error getting team for player {player_id}: {e}")
        #     return 0  # Default to team 0


class FootballStatsTracker:
    def __init__(self):
        self.teams_colors = {0: (0, 0, 255), 1: (255, 0, 0)}  # Default team colors (Blue, Red)
        self.team_classifier = Teams()
        self.ball_positions = None
        self.ball_color = (0, 255, 0)  # Green color for ball
        
        # Statistics
        self.ball_possession = {0: 0, 1: 0}  # Time in frames for each team
        self.passes = {0: 0, 1: 0}  # Number of passes by each team
        self.successful_passes = {0: 0, 1: 0}  # Number of successful passes by each team
        self.failed_passes = {0: 0, 1: 0}  # Number of failed passes by each team
        self.total_frames_with_ball = 0  # Total frames where ball is detected
        
        # Tracking state
        self.last_possession_team = None
        self.last_possession_player = None
        self.possession_threshold = 40  # Pixels - distance to consider a player in possession
        
        # Pass visualization
        self.last_pass_frame = 0  # Track when the last pass was completed for visualization
        self.last_pass_source = None
        self.last_pass_target = None
        
        # Debug mode
        self.debug = True
        self.verbose_debug = False

    def get_bbox_center(self, bbox):
        """Calculate the center of a bounding box"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    def draw_ellipse(self, frame, bbox, color, track_id, team_id, has_ball=False):
        """Draw an ellipse under the player with ID label"""
        # Get the bottom center of the bounding box (player's feet)
        y2 = int(bbox[3])
        x_center = int((bbox[0] + bbox[2]) / 2)
        
        # Calculate ellipse width and height
        width = int(abs(bbox[0] - bbox[2]))
        height = int(0.35 * width)
        
        # Draw the ellipse
        ellipse_color = (0, 255, 255) if has_ball else color  # Highlight player with ball
        cv2.ellipse(frame, (x_center, y2), (width, height), 
                    0, -45, 235, ellipse_color, 2, cv2.LINE_4)
        
        # Draw player ID
        x1_ = x_center - 10
        y1_ = int(y2 + height/2)
        x2_ = x_center + 10
        y2_ = int((y2 + height) + 10)
        
        # Create a small rectangle for the player ID
        cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), ellipse_color, -1)
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (0, 0, 0)  # Black color for text
        text_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            str(track_id), font, font_scale, text_thickness)
        
        # Calculate text position to center it in rectangle
        rect_width = abs(x1_ - x2_)
        rect_height = abs(y1_ - y2_)
        text_x = x1_ + (rect_width - text_width) // 2
        text_y = y1_ + (rect_height + text_height) // 2
        
        # Draw the player ID
        cv2.putText(frame, str(track_id), (text_x, text_y), 
                    font, font_scale, text_color, text_thickness)
        
        return frame

    def draw_ball(self, frame, ball_bbox):
        """Draw the ball with a filled circle and highlight"""
        try:
            if ball_bbox is None:
                return frame
            
            x1, y1, x2, y2 = map(int, ball_bbox)
            
            # Calculate center and radius
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = max(5, int((x2 - x1) / 2))  # Minimum radius of 5 pixels
            
            # Draw a filled circle for the ball
            cv2.circle(frame, (center_x, center_y), radius, self.ball_color, -1)
            
            # Draw a highlight effect (a smaller white circle inside)
            highlight_radius = max(2, int(radius / 3))
            highlight_x = center_x - int(radius * 0.3)
            highlight_y = center_y - int(radius * 0.3)
            cv2.circle(frame, (highlight_x, highlight_y), highlight_radius, (255, 255, 255), -1)
            
            # Draw a thin outline
            cv2.circle(frame, (center_x, center_y), radius + 1, (0, 0, 0), 1)
            
            return frame
        except Exception as e:
            print(f"Error drawing ball: {e}")
            return frame

    def load_ball_positions(self, ball_pickle_path):
        """Load ball positions from pickle file"""
        try:
            print(f"Loading ball positions from {ball_pickle_path}...")
            with open(ball_pickle_path, "rb") as f:
                self.ball_positions = pickle.load(f)
            print(f"Loaded ball positions for {len(self.ball_positions)} frames")
            return True
        except Exception as e:
            print(f"Error loading ball positions: {e}")
            return False

    def find_ball_possession(self, frame, player_tracks, ball_position, frame_idx):
        """Determine which player/team has possession of the ball with improved detection"""
        if ball_position is None:
            return None, None
        
        ball_center = [(ball_position[0] + ball_position[2]) / 2, 
                       (ball_position[1] + ball_position[3]) / 2]
        
        nearest_player = None
        nearest_team = None
        min_distance = float('inf')
        
        # Track all players and their distances to the ball for debugging
        player_distances = []
        
        for player_id, player_data in player_tracks.items():
            if "original_bbox" in player_data:
                bbox = player_data["original_bbox"]
            elif "bbox" in player_data:
                bbox = player_data["bbox"]
            else:
                continue
                
            player_center = self.get_bbox_center(bbox)
            
            # Calculate distance between player and ball
            distance = math.sqrt((player_center[0] - ball_center[0])**2 + 
                                (player_center[1] - ball_center[1])**2)
            
            # Store all player distances for debugging
            if self.debug and self.verbose_debug:
                team_id = self.team_classifier.get_player_team(frame, player_id, bbox)
                player_distances.append((player_id, team_id, distance))
            
            if distance < min_distance:
                min_distance = distance
                nearest_player = player_id
                
                # Get team ID for this player
                nearest_team = self.team_classifier.get_player_team(frame, player_id, bbox)
        
        # Print debugging information about all players' distances from the ball
        if self.debug and self.verbose_debug:
            player_distances.sort(key=lambda x: x[2])  # Sort by distance
            print(f"Frame {frame_idx} - Ball position: {ball_center}")
            for pid, tid, dist in player_distances[:5]:  # Show 5 closest players
                print(f"  Player {pid} (Team {tid}) - Distance: {dist:.2f} px")
            print(f"  Possession threshold: {self.possession_threshold} px")
        
        # Only consider ball in possession if within the threshold distance
        if min_distance <= self.possession_threshold:
            if self.debug and self.verbose_debug:
                print(f"Frame {frame_idx}: Player {nearest_player} (Team {nearest_team}) has possession. Distance: {min_distance:.2f}px")
            return nearest_player, nearest_team
        else:
            if self.debug and self.verbose_debug and min_distance < self.possession_threshold * 2:  # Only show near misses
                print(f"Frame {frame_idx}: No possession. Nearest player {nearest_player} at distance {min_distance:.2f}px")
            return None, None

    def update_player_possession_history(self, frame_idx, player_id, team_id):
        """
        A simplified approach to detect passes directly based on possession changes.
        This method should be called every time we detect ball possession.
        """
        # Skip if no player has possession
        if player_id is None:
            return
            
        # Ensure consistent type for comparison (convert IDs to strings)
        player_id_str = str(player_id) if player_id is not None else None
        last_player_str = str(self.last_possession_player) if self.last_possession_player is not None else None
        
        # Only process if we have valid data (player has possession & we know their team)
        if player_id is not None and team_id is not None and team_id >= 0:
            # Check if this is a different player than the last one with possession
            if (self.last_possession_player is not None and 
                last_player_str != player_id_str):
                
                # If same team, it's a successful pass
                if team_id == self.last_possession_team and self.last_possession_team >= 0:
                    self.passes[team_id] += 1
                    self.successful_passes[team_id] += 1
                    
                    # Record for visualization
                    self.last_pass_frame = frame_idx
                    self.last_pass_source = self.last_possession_player
                    self.last_pass_target = player_id
                    
                    if self.debug:
                        print(f"PASS DETECTED: From player {self.last_possession_player} to player {player_id} (team {team_id}) at frame {frame_idx}")
                
                # If different team, it's a failed pass (interception)
                elif self.last_possession_team is not None and self.last_possession_team >= 0:
                    self.failed_passes[self.last_possession_team] += 1
                    
                    if self.debug:
                        print(f"INTERCEPTION: Ball from player {self.last_possession_player} (team {self.last_possession_team}) intercepted by player {player_id} (team {team_id}) at frame {frame_idx}")
        
        # Always update the last possession information when a player has the ball
        self.last_possession_player = player_id
        self.last_possession_team = team_id

    def draw_statistics(self, frame):
        """Draw ball possession and passes statistics on frame"""
        # Create a semi-transparent overlay for statistics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (460, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Calculate possession percentages
        team0_color = self.teams_colors[0]
        team1_color = self.teams_colors[1]
        
        if self.total_frames_with_ball > 0:
            team0_poss = (self.ball_possession[0] / self.total_frames_with_ball) * 100
            team1_poss = (self.ball_possession[1] / self.total_frames_with_ball) * 100
        else:
            team0_poss = 0
            team1_poss = 0
        
        # Calculate pass success rate
        team0_all_passes = self.successful_passes[0] + self.failed_passes[0]
        team1_all_passes = self.successful_passes[1] + self.failed_passes[1]
        
        team0_success_rate = (self.successful_passes[0] / team0_all_passes * 100) if team0_all_passes > 0 else 0
        team1_success_rate = (self.successful_passes[1] / team1_all_passes * 100) if team1_all_passes > 0 else 0
        
        # Draw team colors as small rectangles
        cv2.rectangle(frame, (20, 70), (40, 90), team0_color, -1)
        cv2.rectangle(frame, (20, 110), (40, 130), team1_color, -1)
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw possession statistics
        cv2.putText(frame, "Ball Possession:", (20, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Team 1: {team0_poss:.1f}%", (50, 85), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Team 2: {team1_poss:.1f}%", (50, 125), font, 0.6, (255, 255, 255), 2)
        
        # Draw passes statistics
        cv2.putText(frame, "Passes:", (20, 160), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Team 1: {self.successful_passes[0]} successful / {self.failed_passes[0]} failed", 
                   (50, 185), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Team 2: {self.successful_passes[1]} successful / {self.failed_passes[1]} failed", 
                   (50, 210), font, 0.6, (255, 255, 255), 2)
        
        # Pass success rate
        cv2.putText(frame, f"Success rate - Team 1: {team0_success_rate:.1f}%, Team 2: {team1_success_rate:.1f}%", 
                   (20, 235), font, 0.6, (255, 255, 255), 2)
        
        return frame

    def annotate_video(self, frames, player_tracks):
        """Annotate video frames with player tracking, ball, and statistics"""
        print("Annotating video frames with players, ball, and statistics...")
        # Process the video
        output_frames = []
        
        # Initialize Teams class
        try:
            self.team_classifier.get_teams_color(player_tracks, frames[0])
            self.teams_colors = self.team_classifier.teams
            print(f"Team colors: {self.teams_colors}")
        except Exception as e:
            print(f"Error initializing team colors: {e}")
        
        # Track possession events for each frame
        possession_events = []
        
        for frame_idx, frame in enumerate(frames):
            frame_idx_str = str(frame_idx)
            if frame_idx_str not in player_tracks:
                # Create a copy with just the frame counter
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)
                
                # Draw statistics
                annotated_frame = self.draw_statistics(annotated_frame)
                
                output_frames.append(annotated_frame)
                continue
                
            annotated_frame = frame.copy()
            
            try:
                # Get ball position for this frame if available
                ball_bbox = None
                if self.ball_positions and frame_idx in self.ball_positions:
                    ball_bbox = self.ball_positions[frame_idx].get('bbox')
                
                # Find which player has the ball
                player_with_ball = None
                if ball_bbox:
                    player_with_ball, team_with_ball = self.find_ball_possession(
                        frame, player_tracks[frame_idx_str]["players"], ball_bbox, frame_idx)
                    
                    # Record possession event for analysis
                    possession_events.append((frame_idx, player_with_ball, team_with_ball))
                    
                    # Use the simplified pass detection approach
                    self.update_player_possession_history(frame_idx, player_with_ball, team_with_ball)
                    
                    # Update possession counter
                    if team_with_ball is not None and team_with_ball >= 0:
                        self.ball_possession[team_with_ball] += 1
                        self.total_frames_with_ball += 1
                
                # Draw ellipses for each player
                for player_id, player_data in player_tracks[frame_idx_str]["players"].items():
                    try:
                        # Check which bbox key is available
                        if "original_bbox" in player_data:
                            bbox = player_data["original_bbox"]
                        elif "bbox" in player_data:
                            bbox = player_data["bbox"]
                        else:
                            continue
                        
                        # Get team ID using Teams class
                        team_id = self.team_classifier.get_player_team(frame, player_id, bbox)
                        
                        # Skip if player is marked as goalkeeper/outlier
                        if team_id == -1:
                            continue
                            
                        # Get team color
                        color = self.teams_colors[team_id]
                        
                        # Check if this player just received a pass
                        just_received_pass = (frame_idx - self.last_pass_frame < 15 and 
                                              player_id == self.last_pass_target)
                        
                        # Draw ellipse and player ID
                        has_ball = (player_id == player_with_ball)
                        ellipse_color = (0, 255, 255) if has_ball else color  
                        if just_received_pass:
                            ellipse_color = (255, 255, 0)  # Yellow for player who just received a pass
                            
                        annotated_frame = self.draw_ellipse(annotated_frame, bbox, color, player_id, team_id, has_ball)
                        
                        # Draw pass indicator if this player just received a pass
                        if just_received_pass and self.last_pass_source is not None:
                            # Find the source player position if they're visible
                            source_player_pos = None
                            if str(self.last_pass_source) in player_tracks[frame_idx_str]["players"]:
                                source_data = player_tracks[frame_idx_str]["players"][str(self.last_pass_source)]
                                if "original_bbox" in source_data:
                                    source_bbox = source_data["original_bbox"]
                                    source_player_pos = self.get_bbox_center(source_bbox)
                                elif "bbox" in source_data:
                                    source_bbox = source_data["bbox"]
                                    source_player_pos = self.get_bbox_center(source_bbox)
                            
                            if source_player_pos:
                                player_center = self.get_bbox_center(bbox)
                                # Draw a dashed line showing the pass
                                pt1 = (int(source_player_pos[0]), int(source_player_pos[1]))
                                pt2 = (int(player_center[0]), int(player_center[1]))
                                
                                # Draw dashed line
                                dash_length = 10
                                dash_gap = 5
                                d = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                                if d > 0:
                                    dash_count = int(d / (dash_length + dash_gap))
                                    for i in range(dash_count):
                                        start_ratio = i * (dash_length + dash_gap) / d
                                        end_ratio = start_ratio + dash_length / d
                                        if end_ratio > 1:
                                            end_ratio = 1
                                            
                                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
                                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
                                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
                                        
                                        cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), 
                                                (255, 255, 0), 2)
                                
                                # Label the pass
                                mid_x = int((source_player_pos[0] + player_center[0]) / 2)
                                mid_y = int((source_player_pos[1] + player_center[1]) / 2)
                                cv2.putText(annotated_frame, "PASS", (mid_x, mid_y - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                
                    except Exception as e:
                        print(f"Error annotating player {player_id} in frame {frame_idx}: {e}")
                        continue
                
                # Draw the ball if available for this frame
                if ball_bbox:
                    annotated_frame = self.draw_ball(annotated_frame, ball_bbox)
                
                # Draw statistics
                annotated_frame = self.draw_statistics(annotated_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            
            output_frames.append(annotated_frame)
            
            # Print progress every 100 frames
            if frame_idx % 100 == 0:
                print(f"Annotated frame {frame_idx}")
                
        return output_frames

    def configure_pass_parameters(self, possession_threshold=40, pass_max_duration=60):
        """Configure parameters for pass detection"""
        self.possession_threshold = possession_threshold  # Distance to consider a player in possession
        print(f"Pass parameters configured: possession_threshold={possession_threshold}")

    def reset_pass_statistics(self):
        """Reset all pass-related statistics for testing"""
        self.passes = {0: 0, 1: 0}
        self.successful_passes = {0: 0, 1: 0}
        self.failed_passes = {0: 0, 1: 0}
        self.last_pass_frame = 0
        self.last_pass_source = None
        self.last_pass_target = None
        print("Pass statistics have been reset")
    
    def enhance_debug_mode(self, enable=True, verbose=False):
        """Configure debug mode and verbosity level"""
        self.debug = enable
        self.verbose_debug = verbose if enable else False
        print(f"Debug mode {'enabled' if enable else 'disabled'}, verbose: {verbose}")
    
    def calculate_pass_metrics(self):
        """Calculate advanced pass metrics based on collected statistics"""
        metrics = {}
        
        for team_id in [0, 1]:
            total_team_passes = self.successful_passes[team_id] + self.failed_passes[team_id]
            completion_rate = (self.successful_passes[team_id] / total_team_passes * 100) if total_team_passes > 0 else 0
            
            metrics[f'team{team_id+1}'] = {
                'total_passes': total_team_passes,
                'successful_passes': self.successful_passes[team_id],
                'failed_passes': self.failed_passes[team_id],
                'completion_rate': completion_rate,
                'possession_percentage': (self.ball_possession[team_id] / self.total_frames_with_ball * 100) 
                                        if self.total_frames_with_ball > 0 else 0
            }
        
        # Calculate the ratio of passes to possession time
        for team_id in [0, 1]:
            possession_frames = self.ball_possession[team_id]
            total_passes = self.successful_passes[team_id] + self.failed_passes[team_id]
            
            # Passes per 100 frames of possession
            if possession_frames > 0:
                metrics[f'team{team_id+1}']['passes_per_possession'] = (total_passes / possession_frames) * 100
            else:
                metrics[f'team{team_id+1}']['passes_per_possession'] = 0
        
        return metrics
        
    def process_video(self, video_path, output_path, player_tracks_path, ball_positions_path=None, color_threshold=100):
        """Process full video with player tracking, ball annotation, and statistics"""
        # Set color threshold for goalkeeper detection
        self.team_classifier.color_threshold = color_threshold
        
        # Load ball positions if path is provided
        if ball_positions_path:
            self.load_ball_positions(ball_positions_path)
        
        # Load video
        print(f"Loading video from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a VideoWriter object if output path is provided
        out = None
        if output_path:
            print(f"Output will be saved to {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Read all frames
        print("Reading frames...")
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
            if len(frames) % 100 == 0:
                print(f"Read {len(frames)} frames")
                
        cap.release()
        print(f"Loaded {len(frames)} frames from video")
        
        # Read player tracks from pickle file
        print(f"Reading player tracks from {player_tracks_path}...")
        with open(player_tracks_path, 'rb') as f:
            tracks = pickle.load(f)
        
        # Print sample of data structure for debugging
        print("Sample of tracks data structure:")
        first_key = list(tracks.keys())[0]
        print(f"Frame {first_key} contains: {list(tracks[first_key].keys())}")
        if "players" in tracks[first_key]:
            player_keys = list(tracks[first_key]["players"].keys())
            if player_keys:
                first_player = player_keys[0]
                print(f"First player data: {tracks[first_key]['players'][first_player]}")
        
        # Convert frame indices to strings if they're not already
        if all(isinstance(k, int) for k in tracks.keys()):
            print("Converting frame indices from integers to strings...")
            tracks = {str(k): v for k, v in tracks.items()}
        
        # Print debugging information about the tracking data
        if self.debug:
            print(f"Debugging information:")
            print(f"  Total frames in video: {len(frames)}")
            print(f"  Total frames with tracking data: {len(tracks)}")
            if self.ball_positions:
                print(f"  Total frames with ball positions: {len(self.ball_positions)}")
        
        # Annotate video with players, ball, and statistics
        output_frames = self.annotate_video(frames, tracks)
        
        # Write output video if requested
        if out is not None and output_frames:
            print("Writing output video...")
            for frame in output_frames:
                out.write(frame)
            out.release()
            print(f"Output saved to {output_path}")
            
        # Calculate advanced metrics
        advanced_metrics = self.calculate_pass_metrics()
        
        # Print final statistics
        team0_poss = (self.ball_possession[0] / self.total_frames_with_ball * 100) if self.total_frames_with_ball > 0 else 0
        team1_poss = (self.ball_possession[1] / self.total_frames_with_ball * 100) if self.total_frames_with_ball > 0 else 0
        
        # Calculate success rates
        team0_all_passes = self.successful_passes[0] + self.failed_passes[0]
        team1_all_passes = self.successful_passes[1] + self.failed_passes[1]
        
        team0_success_rate = (self.successful_passes[0] / team0_all_passes * 100) if team0_all_passes > 0 else 0
        team1_success_rate = (self.successful_passes[1] / team1_all_passes * 100) if team1_all_passes > 0 else 0
        
        print("\n==== FINAL STATISTICS ====")
        print(f"Ball Possession - Team 1: {team0_poss:.1f}%, Team 2: {team1_poss:.1f}%")
        print(f"Passes - Team 1: {self.successful_passes[0]} successful, {self.failed_passes[0]} failed ({team0_success_rate:.1f}% success rate)")
        print(f"Passes - Team 2: {self.successful_passes[1]} successful, {self.failed_passes[1]} failed ({team1_success_rate:.1f}% success rate)")
        
        # Print advanced metrics
        print("\n==== ADVANCED METRICS ====")
        print(f"Team 1 Passes per 100 frames of possession: {advanced_metrics['team1']['passes_per_possession']:.2f}")
        print(f"Team 2 Passes per 100 frames of possession: {advanced_metrics['team2']['passes_per_possession']:.2f}")
        
        # Return statistics alongside the video
        stats = {
            'possession': {
                'team1': team0_poss,
                'team2': team1_poss
            },
            'passes': {
                'team1': {
                    'successful': self.successful_passes[0],
                    'failed': self.failed_passes[0],
                    'total': team0_all_passes,
                    'success_rate': team0_success_rate
                },
                'team2': {
                    'successful': self.successful_passes[1],
                    'failed': self.failed_passes[1],
                    'total': team1_all_passes,
                    'success_rate': team1_success_rate
                }
            },
            'advanced_metrics': advanced_metrics
        }
        
        return output_frames, stats


def read_video_frames(video_path):
    """Helper function to read frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    return frames



if __name__ == "__main__":
    # Initialize tracker
    tracker = FootballStatsTracker()
    
    # Enable debug mode for better problem diagnosis
    tracker.enhance_debug_mode(enable=True, verbose=False)
    
    # Configure pass parameters with more generous values for better detection
    tracker.configure_pass_parameters(possession_threshold=40)
    
    # Process the video with player tracking, ball annotation, and pass success statistics
    output_frames, stats = tracker.process_video(
        video_path="/home/elmehdi/Desktop/footballe_analyseur/videos/test9.mp4",
        output_path="football_stats_with_pass_tracking.mp4",
        player_tracks_path="/home/elmehdi/Desktop/footballe_analyseur/updated_football_all_frames.pkl",
        ball_positions_path="/home/elmehdi/Desktop/footballe_analyseur/ball_positions.pkl",
        color_threshold=100
    )
    
    # Print the final statistics
    print("\n==== FINAL STATISTICS ====")
    print(f"Stats summary: {stats}")