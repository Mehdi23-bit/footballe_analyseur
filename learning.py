from utilis.utilis import read ,output
from Tracker.Tracker import Tracker
import time
import os
start=time.time()
if os.path.exists('final_videos/Proccessed_game1337.avi'):
    os.remove('final_videos/Proccessed_game1337.avi')
frames=read('videos/test30.mp4')
tracker=Tracker('models/best1.pt')
tracks=tracker.get_object_tracks(frames,False,'tracks_pickles/track1.pkl')
tracks["ball"]=tracker.interpolate_ball_positions(tracks["ball"])
output_frame=tracker.draw_annotation('videos/test14.mp4',tracks)

output(output_frame,20,'final_videos/Proccessed_game1337.avi')
print(time.time()-start)
