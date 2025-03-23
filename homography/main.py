import cv2
import supervision as sv
import numpy as np
from inference import get_model
from sports.configs.soccer import SoccerPitchConfiguration
from view import ViewTransformer
from utilis import read
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

frames = read("test9.mp4")
ROBOFLOW_API_KEY = "ogr5YVLXS9inrOrdCp5t"
SOURCE_IMAGE_PATH = "/content/first_frame.jpg"

player_detection_model = get_model(
    model_id="football-players-detection-3zvbc/12",
    api_key=ROBOFLOW_API_KEY
)

pitch_detection_model = get_model(
    model_id="football-field-detection-f07vi/15",
    api_key=ROBOFLOW_API_KEY
)

image = cv2.imread(SOURCE_IMAGE_PATH)

def process_frame(frame):
    field_result = pitch_detection_model.infer(frame, confidence=0.3)[0]
    keypoints = sv.KeyPoints.from_inference(field_result)
    
    filter = keypoints.confidence[0] > 0.5
    
    CONFIG = SoccerPitchConfiguration()
    transformer = ViewTransformer(
        source=keypoints.xy[0][filter].astype(np.float32),
        target=np.array(CONFIG.vertices)[filter].astype(np.float32)
    )
    
    player_result = player_detection_model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(player_result)
    
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)
    
    return detections, transformed_xy, transformer, CONFIG

def draw_pitch(ax, config, scale_factor=0.1, mirror_vertical=True):
    """Draw soccer pitch on matplotlib axes with scaled dimensions"""
    # Scale dimensions
    width = config.width * scale_factor
    length = config.length * scale_factor
    penalty_box_width = config.penalty_box_width * scale_factor
    penalty_box_length = config.penalty_box_length * scale_factor
    goal_box_width = config.goal_box_width * scale_factor
    goal_box_length = config.goal_box_length * scale_factor
    centre_circle_radius = config.centre_circle_radius * scale_factor
    
    # When mirroring vertically, flip the y-coordinates
    if mirror_vertical:
        # Field outline
        ax.plot([0, 0], [width, 0], 'white')
        ax.plot([0, length], [width, width], 'white')
        ax.plot([length, length], [width, 0], 'white')
        ax.plot([0, length], [0, 0], 'white')
        
        # Halfway line
        ax.plot([length/2, length/2], [width, 0], 'white')
        
        # Center circle
        center_circle = Circle((length/2, width/2), centre_circle_radius, fill=False, color='white')
        ax.add_patch(center_circle)
        
        # Left penalty area
        ax.plot([0, penalty_box_length], [width-(width-penalty_box_width)/2, width-(width-penalty_box_width)/2], 'white')
        ax.plot([penalty_box_length, penalty_box_length], [width-(width-penalty_box_width)/2, width-(width+penalty_box_width)/2], 'white')
        ax.plot([0, penalty_box_length], [width-(width+penalty_box_width)/2, width-(width+penalty_box_width)/2], 'white')
        
        # Right penalty area
        ax.plot([length, length-penalty_box_length], [width-(width-penalty_box_width)/2, width-(width-penalty_box_width)/2], 'white')
        ax.plot([length-penalty_box_length, length-penalty_box_length], [width-(width-penalty_box_width)/2, width-(width+penalty_box_width)/2], 'white')
        ax.plot([length, length-penalty_box_length], [width-(width+penalty_box_width)/2, width-(width+penalty_box_width)/2], 'white')
        
        # Left goal area
        ax.plot([0, goal_box_length], [width-(width-goal_box_width)/2, width-(width-goal_box_width)/2], 'white')
        ax.plot([goal_box_length, goal_box_length], [width-(width-goal_box_width)/2, width-(width+goal_box_width)/2], 'white')
        ax.plot([0, goal_box_length], [width-(width+goal_box_width)/2, width-(width+goal_box_width)/2], 'white')
        
        # Right goal area
        ax.plot([length, length-goal_box_length], [width-(width-goal_box_width)/2, width-(width-goal_box_width)/2], 'white')
        ax.plot([length-goal_box_length, length-goal_box_length], [width-(width-goal_box_width)/2, width-(width+goal_box_width)/2], 'white')
        ax.plot([length, length-goal_box_length], [width-(width+goal_box_width)/2, width-(width+goal_box_width)/2], 'white')
    else:
        # Original non-mirrored version
        # Field outline
        ax.plot([0, 0], [0, width], 'white')
        ax.plot([0, length], [0, 0], 'white')
        ax.plot([length, length], [0, width], 'white')
        ax.plot([0, length], [width, width], 'white')
        
        # Halfway line
        ax.plot([length/2, length/2], [0, width], 'white')
        
        # Center circle
        center_circle = Circle((length/2, width/2), centre_circle_radius, fill=False, color='white')
        ax.add_patch(center_circle)
        
        # Left penalty area
        ax.plot([0, penalty_box_length], [(width-penalty_box_width)/2, (width-penalty_box_width)/2], 'white')
        ax.plot([penalty_box_length, penalty_box_length], [(width-penalty_box_width)/2, (width+penalty_box_width)/2], 'white')
        ax.plot([0, penalty_box_length], [(width+penalty_box_width)/2, (width+penalty_box_width)/2], 'white')
        
        # Right penalty area
        ax.plot([length, length-penalty_box_length], [(width-penalty_box_width)/2, (width-penalty_box_width)/2], 'white')
        ax.plot([length-penalty_box_length, length-penalty_box_length], [(width-penalty_box_width)/2, (width+penalty_box_width)/2], 'white')
        ax.plot([length, length-penalty_box_length], [(width+penalty_box_width)/2, (width+penalty_box_width)/2], 'white')
        
        # Left goal area
        ax.plot([0, goal_box_length], [(width-goal_box_width)/2, (width-goal_box_width)/2], 'white')
        ax.plot([goal_box_length, goal_box_length], [(width-goal_box_width)/2, (width+goal_box_width)/2], 'white')
        ax.plot([0, goal_box_length], [(width+goal_box_width)/2, (width+goal_box_width)/2], 'white')
        
        # Right goal area
        ax.plot([length, length-goal_box_length], [(width-goal_box_width)/2, (width-goal_box_width)/2], 'white')
        ax.plot([length-goal_box_length, length-goal_box_length], [(width-goal_box_width)/2, (width+goal_box_width)/2], 'white')
        ax.plot([length, length-goal_box_length], [(width+goal_box_width)/2, (width+goal_box_width)/2], 'white')
    
    # Set limits and background
    ax.set_xlim(-5, length+5)
    ax.set_ylim(-5, width+5)
    ax.set_facecolor('green')
    
    return ax

def display_players_on_pitch(detections, transformed_xy, config, frame=None, scale_factor=0.1, mirror_vertical=True):
    """Display players on a scaled soccer pitch"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Draw the pitch on the first axis
    draw_pitch(ax1, config, scale_factor, mirror_vertical)
    
    # Plot players on the pitch
    team1_color = 'red'
    team2_color = 'blue'
    
    # Assign teams based on x-position (left/right side of field)
    for i, (x, y) in enumerate(transformed_xy):
        x_scaled = x * scale_factor
        # Mirror y-coordinate if requested
        if mirror_vertical:
            y_scaled = (config.width - y) * scale_factor
        else:
            y_scaled = y * scale_factor
        
        # Simple team assignment based on x-position
        # You may need a more sophisticated approach
        if x < config.length / 2:
            color = team1_color
        else:
            color = team2_color
            
        player_circle = Circle((x_scaled, y_scaled), 30 * scale_factor, color=color, alpha=0.7)
        ax1.add_patch(player_circle)
        ax1.text(x_scaled, y_scaled, str(i), color='white', ha='center', va='center', fontsize=8)
    
    # Show original frame with detections
    if frame is not None:
        # Draw bounding boxes on the original frame
        annotated_frame = frame.copy()
        
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Simple team assignment based on transformed position
            if transformed_xy[i][0] < config.length / 2:
                color = (0, 0, 255)  # BGR red
            else:
                color = (255, 0, 0)  # BGR blue
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, str(i), (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert BGR to RGB for matplotlib
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        ax2.imshow(annotated_frame_rgb)
        ax2.set_title("Original Frame with Detections")
        ax2.axis('off')
    
    ax1.set_title("Players on Pitch")
    ax1.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def process_video_frame(frame_idx):
    """Process a specific frame from the video"""
    if frame_idx < len(frames):
        frame = frames[frame_idx]
        detections, transformed_xy, transformer, config = process_frame(frame)
        
        # Display players on the pitch
        fig = display_players_on_pitch(detections, transformed_xy, config, frame, scale_factor=0.1)
        
        # Save or display the figure
        plt.savefig(f"frame_{frame_idx}_visualization.png")
        plt.close(fig)
        
        return detections, transformed_xy, transformer
    else:
        print(f"Frame index {frame_idx} out of range. Video has {len(frames)} frames.")
        return None, None, None

# Process the first frame from the image
detections, transformed_xy, transformer, config = process_frame(image)

# Create visualization of the first frame
fig = display_players_on_pitch(detections, transformed_xy, config, image, scale_factor=0.1)
plt.savefig("first_frame_visualization.png")
plt.show()

# Example: Process frame 50 from the video
# video_detections, video_transformed_xy, video_transformer = process_video_frame(50)