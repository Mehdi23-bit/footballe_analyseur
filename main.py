from utilis.utilis import read ,output
from Tracker.Tracker import Tracker
import time
import os
from Camera.camera import Camera 
import pickle
file_path = "final_videos/Proccessed_game1337.avi"  

if os.path.exists(file_path):
    os.remove(file_path)
start=time.time()
frames=read('videos/test9.mp4')
tracker=Tracker('models/best1.pt')
tracks=tracker.get_object_tracks(frames,True,'tracks_pickles/track.pkl')
tracks["ball"]=tracker.interpolate_ball_positions(tracks["ball"])

with open("tracks_pickles/player0.pkl",'wb') as f:
                pickle.dump(tracks["players"],f)
output_frame=tracker.draw_annotation('videos/test9.mp4',tracks)
camera=Camera()   
camera_mv=camera.get_camera_mvn("videos/test9.mp4")
output_frame=camera.draw_camera_mvnt(output_frame)
player_tracks=tracker.adjust_player_movent(tracks["players"],camera_mv)
with open("tracks_pickles/player1.pkl",'wb') as f:
    pickle.dump(player_tracks,f)

output(output_frame,20,'final_videos/Proccessed_game1337.avi')
print(time.time()-start)


