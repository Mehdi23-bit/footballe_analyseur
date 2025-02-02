from utilis.utilis import read ,output
from Tracker.Tracker import Tracker
frames=read('videos/test14.mp4')
tracker=Tracker('models/best.pt')
tracks=tracker.get_object_tracks(frames,True,'tracks_pickles/track_test.pkl')

output_frame=tracker.draw_annotation('videos/test14.mp4',tracks)
output(output_frame,20,'final_videos/Proccessed_game.avi')