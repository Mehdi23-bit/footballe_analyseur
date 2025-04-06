import cv2
from utilis.utilis import read
import math
import pickle
import os
class Camera:
    def __init__(self):
        self.path="tracks_pickles/camera.pkl"
   
    def get_camera_mvn(self,video_path):
       
        if os.path.exists(self.path):
            with open(self.path,'rb') as f:
                tracks = pickle.load(f)
                self.camera=tracks
            return tracks
        frames=read(video_path)
        features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
             )

        self.camera=[(0,0)]*len(frames)
       
        for frame_num in range(len(frames)-1):
            
            gray_frame0 = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            gray_frame1=cv2.cvtColor(frames[frame_num+1], cv2.COLOR_BGR2GRAY)
            features = cv2.goodFeaturesToTrack(gray_frame0, maxCorners=100, qualityLevel=0.3, minDistance=7)
            next_points, status, error = cv2.calcOpticalFlowPyrLK(gray_frame0,gray_frame1, features, None)
            max_distace=0
            max_x=0
            max_y=0
            for i, (new_point, old_point) in enumerate(zip(next_points, features)):
                x, y = new_point.ravel() 
                x_,y_=old_point.ravel()
                distance=math.sqrt((x-x_)**2+(y-y_)**2)
                if distance>max_distace:
                     max_distace= distance
                     max_x=x-x_
                     max_y=y-y_
            if max_distace>5:         
                self.camera[frame_num]=(max_x,max_y)
                
        
        with open(self.path,'wb') as f:
              pickle.dump(self.camera,f)              
        return self.camera    

    def draw_camera_mvnt(self, frames): 
        for i, frame in enumerate(frames):
            overlay = frame.copy()
            cv2.rectangle(frame, (0,900),(500,1000),(255,255,255), thickness=-1)  
            
            # Corrected cv2.addWeighted usage
            blended = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)  # Use proper alpha, beta, and gamma
           
            # Put text on blended frame
           
            cv2.putText(blended, f"X={self.camera[i][0]:.2f}", (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(blended, f"Y={self.camera[i][1]:.2f}", (10, 980), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) 
            frames[i] = blended  # Store the updated frame

        return frames       
