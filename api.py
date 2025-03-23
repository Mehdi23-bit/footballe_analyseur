from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create necessary directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "final_videos"
PICKLE_DIR = "tracks_pickles"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)



logger = logging.getLogger(__name__)

def convert_avi_to_mp4(avi_file_path, output_mp4_path=None, fps=30.0, resize=None):
    try:
        import ffmpeg
        # Input and output file names
        input_file = avi_file_path
        output_file = output_mp4_path

        # Run FFmpeg with web-compatible settings
        ffmpeg.input(input_file).output(
            output_file, 
            vcodec='libx264',
            preset='fast',
            crf=23,
            pix_fmt='yuv420p',       # Important for browser compatibility
            movflags='+faststart'     # Optimizes for web streaming
        ).run()
        
        # Verify the output file exists
        if os.path.exists(output_mp4_path) and os.path.getsize(output_mp4_path) > 0:
            print(f"Successfully converted to MP4: {output_mp4_path}")
            return output_mp4_path
        else:
            print("Conversion failed: output file is missing or empty")
            return None
    except Exception as e:
        print(f"Error during AVI to MP4 conversion: {str(e)}")
        return None
    
def process_video(input_path, unique_id):
    print("processing 1 2 3")
    start = time.time()
    
    # Check if input video file exists
    if not os.path.exists(input_path):
        print(f"Error: Input video file '{input_path}' not found!")
        return None, None
    
    # Check if model file exists
    model_path = '/home/elmehdi/Desktop/footballe_analyseur/models/best1.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return None, None
    else:
        print("model found")
        
    # Check if output directory exists, create it if not
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' does not exist. Creating it.")
        os.makedirs(OUTPUT_DIR)
        
    print("read frames")
    # Read video frames
    frames = read(input_path)
    
    print("Load model")
    # Initialize tracker and get tracks
    tracker = Tracker(model_path)
    print("load model301")
    tracks = tracker.get_object_tracks(frames, True, f"{PICKLE_DIR}/track.pkl")
    
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("Interpolate ball")

    # Save player tracks
    with open(f"{PICKLE_DIR}/player0.pkl", 'wb') as f:
        pickle.dump(tracks["players"], f)

    # Draw annotations
    output_frame = tracker.draw_annotation(input_path, tracks)
    
    # Process camera movement
    camera = Camera()
    camera_mv = camera.get_camera_mvn(input_path)
    output_frame = camera.draw_camera_mvnt(output_frame)
    
    # Adjust player movement
    player_tracks = tracker.adjust_player_movent(tracks["players"], camera_mv)
    
    # Save adjusted player tracks
    with open(f"{PICKLE_DIR}/player1_{unique_id}.pkl", 'wb') as f:
        pickle.dump(player_tracks, f)
    
    print("Save video")
    print(unique_id)
    
    # Generate output video (only once)
    avi_output_path = f"{OUTPUT_DIR}/Processed_{unique_id}.avi"
    output(output_frame, 20, avi_output_path)
    
    # Calculate processing time for AVI
    avi_processing_time = time.time() - start
    print(f"AVI Processing completed in {avi_processing_time:.2f} seconds")
    
    # Try to convert to MP4 using OpenCV
    mp4_output_path = f"{OUTPUT_DIR}/Processed_{unique_id}.mp4"
    try:
        print(f"Converting AVI to MP4 using OpenCV: {avi_output_path} -> {mp4_output_path}")
        converted_path = convert_avi_to_mp4(avi_output_path, mp4_output_path, fps=20)
        
        if converted_path and os.path.exists(converted_path):
            print("MP4 conversion successful")
            total_processing_time = time.time() - start
            print(f"Total processing time: {total_processing_time:.2f} seconds")
            return mp4_output_path, total_processing_time
        else:
            print("MP4 conversion failed, falling back to AVI")
            return avi_output_path, avi_processing_time
    except Exception as e:
        print(f"Error during video conversion: {str(e)}")
        # Fall back to AVI if conversion fails
        return avi_output_path, avi_processing_time

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    print('i am uploading')
    print(file.filename)
    # Generate unique filename to avoid collisions
    unique_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    
    # Save uploaded file
    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}{file_extension}")
    print(input_path)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the video immediately (synchronously)
    try:
        print("before processing")
        output_path, processing_time = process_video(input_path, unique_id)  # Remove file extension here
        print("end processing")
        
        # Check if file exists and is readable
        if not os.path.exists(output_path):
            print(f"ERROR: Output file does not exist: {output_path}")
            return {"error": "Output file not found", "path": output_path}
            
        # Get absolute path - this might be necessary
        abs_output_path = os.path.abspath(output_path)
        print(f"Absolute output path: {abs_output_path}")
        
        # Check file size and permissions
        file_size = os.path.getsize(abs_output_path)
        print(f"File size: {file_size} bytes")
        
        # Try to read a small portion to verify access
        try:
            with open(abs_output_path, 'rb') as test_file:
                test_file.read(1024)
            print("File can be read successfully")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return {"error": f"File access error: {str(e)}"}
        
        # Return the processed video file directly
        print("returning file")
        return FileResponse(
            path=abs_output_path,  # Use absolute path here
            filename=f"Processed_{original_filename}",
            media_type="video/x-msvideo"  # for .avi files
        )
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {
            "error": f"Processing failed: {str(e)}",
            "filename": original_filename,
            "id": unique_id
        }

@app.get("/")
async def hello():
    # Path to your video file
    video_path = "/home/elmehdi/Desktop/footballe_analyseur/final_videos/Processed_399dbede-c2c7-49e4-b8e1-5282ac5f43c3.avi"
    
    # Check if the file exists
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}
    
    # Return the video file
    return FileResponse(
        path=video_path,
        media_type="video/x-msvideo",  # For .avi files
        filename="football_analysis.avi"  # This is the name shown to users when downloading
    )