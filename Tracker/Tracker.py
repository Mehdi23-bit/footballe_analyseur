from  ultralytics import YOLO  
import cv2
import supervision as sv
import pickle
from utilis.utilis import read
import numpy as np
from sklearn.cluster import KMeans
from team import  Teams
import pandas as pd
import math
class Tracker:
    def __init__(self,model_p):
        self.model=YOLO(model_p)
        self.tracker=sv.ByteTrack()
        self.is_the_same=None
        self.passes=0
        self.teams={0:0,1:0}

    def adjust_player_movent(self,tracks,camera_movments):
             for i,frame_num in enumerate(tracks):
                print(i)
                for player in frame_num:
                    frame_num[player]["bbox"][0]-=camera_movments[i][0]
                    frame_num[player]["bbox"][1]-=camera_movments[i][1]
                    frame_num[player]["bbox"][2]-=camera_movments[i][0]
                    frame_num[player]["bbox"][3]-=camera_movments[i][1]
             return tracks            

    def is_between(self,num1,num2,num3):
        return  num1>=num2 and num1<=num3
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

       


    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            print(detections)
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
    

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub:
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        print("detections are : " + detections)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # # Convert GoalKeeper to player object
            # for object_ind , class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    def draw_ellipse(self,frame,bbox,color,track_id,text,ball):
        if ball:
            text=str(1337)
        y2=int(bbox[3])
        x_center=int((bbox[0]+bbox[2])/2)
        width=int(abs(bbox[0]-bbox[2]))
        height= int(0.35 * width)
        x1_=x_center-10
        y1_=int(y2+height/2)
        x2_=x_center+10
        y2_=int((y2+height)+10)
        cv2.ellipse(frame, (x_center, y2), (width,height), 
            0, -45, 235, color, 2, cv2.LINE_4)
        cv2.rectangle(frame, (x1_,y1_),(x2_,y2_), color, -1)
        x_txt_cnt=int((x1_+x2_)/2)
        y_txt_cnt=int((y1_+y2_)/2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (0, 0, 0)  # Red color in BGR
        text_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

# Calculate the center of the rectangle
        rect_width = abs(x1_-x2_)
        rect_height = abs(y1_-y2_)
# Compute text position to center it in the rectangle
        text_x = x1_ + (rect_width - text_width) // 2
        text_y = y1_ + (rect_height + text_height) // 2
        cv2.putText(frame, str(track_id), (text_x, text_y), font, font_scale, text_color, text_thickness)

        return frame

    def draw_traingle(self,frame,bbx,color,width,height):
        x_center=int((bbx[0]+bbx[2])/2)
        y_center=int(bbx[1])
        triangle_points = np.array([[x_center-width,y_center-height], [x_center,y_center], [x_center+width, y_center-height]], np.int32) 
        triangle_points = triangle_points.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [triangle_points], color=color, lineType=cv2.LINE_AA)
              
        return frame
    
    
    def extract_jersey_region(self,image,bbx):
        x1, y1, x2, y2 =  map(int, bbx)  
        
        jersey_region = image[y1:y2, x1:x2]  
        if jersey_region is None:
            raise ValueError("extract_jersey_region returned None")
        if not isinstance(jersey_region, np.ndarray):
            raise ValueError(f"extract_jersey_region returned {type(jersey_region)} instead of np.ndarray")
        if jersey_region.size == 0:
            raise ValueError("extract_jersey_region returned an empty array")   
        return jersey_region 

    def get_dominant_color(self, image,bbox, k=2):
    # Ensure image is a NumPy array
        image = np.array(image)
        x_center=int((bbox[0]+bbox[2])/2)
        y_center=int((bbox[1]+bbox[3])/2)
        pixel_color = image[y_center, x_center]
        print(pixel_color)
        return  tuple(map(int, pixel_color)) 
        
       
    def draw_annotation(self,video,tracks):
       participant={}
       video_frames=read(video)
       players_dict=tracks['players'] 
       referers_dict=tracks['referees']
       balls=tracks['ball'] 
       outputvidep=[]
       team_obj=Teams()
       who_has_ball=None
       who_has_ball_bbox=None
       noneclr=None
       holder_team=None
       for frame_num,frame in enumerate(video_frames):  
           frame=frame.copy()
           is_already=False
           for track_id,player in players_dict[frame_num].items():
               is_had_ball=False
               team_clr=team_obj.get_teams_color(tracks,frame)
               id=team_obj.get_player_team(frame,track_id,player["bbox"])
               player_team_clr=team_clr[id]
               x1,y1,x2,y2=player["bbox"][0],player["bbox"][1],player["bbox"][2],player["bbox"][3]
               if balls[frame_num][1]:
                bx1,by1,bx2,by2=balls[frame_num][1]["bbox"]
               ball_mid=[int((bx2+bx1)/2),int((by2+by1)/2)]
               distance1=math.sqrt((x1-ball_mid[0])**2+(y2-ball_mid[1])**2)
               distance2=math.sqrt((x2-ball_mid[0])**2+(y2-ball_mid[1])**2)
               distance=min(distance1,distance2)
               print(distance)
               if distance<20:
                   participant[track_id]=distance
                       
               frame=self.draw_ellipse(frame,player["bbox"],player_team_clr,track_id,str(id),is_had_ball) 
              

               
           print(participant)
           
           if participant: 
            who_has_ball=min(participant, key=participant.get)
            who_has_ball_bbox=participant[who_has_ball]
            

            print(f"min is {who_has_ball}")
            p_bbox=players_dict[frame_num][who_has_ball]['bbox']
            frame=self.draw_traingle(frame,p_bbox,(0,0,255),5,10)
            participant={}
           if who_has_ball_bbox is None:
                noneclr=(0,0,0)  
           else:
             if self.is_the_same!=who_has_ball:
                self.passes+=1
                holder_team=team_obj.get_player_team(frame,who_has_ball,who_has_ball_bbox)  
                self.teams[holder_team]+=1
                    

                 
           self.is_the_same=who_has_ball
           top_left = (100, 100)
           bottom_right = (550, 400)
           color = (255, 255,255)  # Rectangle color (green in BGR)
           thickness = 3  # Line thickness

                # Draw the rectangle
           overlay=frame.copy()     
           cv2.rectangle(frame, (0,0), (450,100), color, thickness=-1)
           cv2.addWeighted(overlay, 0.5, frame, 0.5, 0,frame)
                # Define text parameters: position, text, font, font size, color, and thickness
           text_total = f"total passes {str(self.passes)}"
           text_total1 = f"total passes of team's 0: {str(self.teams[0])}"
           text_total2 = f"total passes of team's 1: {str(self.teams[1])}"
           font = cv2.FONT_HERSHEY_SIMPLEX
           font_size = 1
           if holder_team is None:
            text_color=noneclr
            
           else:
               text_color = team_obj.teams[holder_team] 
               print(f"i am the holding team {holder_team}")
                # Text color (white in BGR)
           text_thickness = 2
           text_position = (150, 250)  # Position to start the text

           overlay=frame.copy()
           cv2.rectangle(frame, (1400,900),(1900,1000),color, thickness=-1)
           cv2.addWeighted(overlay, 0.5, frame, 0.5, 0,frame)
           try:
            possession1=round((self.teams[0]/(self.teams[0]+self.teams[1]))*100,2) 
            possession2=round((self.teams[1]/(self.teams[0]+self.teams[1]))*100,2) 
    
           except ZeroDivisionError:
               possession1=0
               possession2=0
           poss1=f"team's 0 possession is :{possession1}%"
           poss2=f"team's 1 possession is :{possession2}%"
           cv2.putText(frame,poss1, (1400,950), font, font_size, (0,0,0), text_thickness) 
           cv2.putText(frame,poss2, (1400,1000), font, font_size, (0,0,0), text_thickness) 
           cv2.putText(frame, text_total, (0, 30), font, font_size, (0,0,0), text_thickness)
           cv2.putText(frame, text_total1, (0, 60), font, font_size, (0,0,0), text_thickness)
           cv2.putText(frame, text_total2, (0, 90), font, font_size, (0,0,0), text_thickness)    

 
           for track_id,referee in referers_dict[frame_num].items():
               frame=self.draw_ellipse(frame,referee["bbox"],(0,255,255),track_id,str(track_id),False)
           for track_id,ball in balls[frame_num].items():
               frame=self.draw_traingle(frame,ball['bbox'],(0,255,0),5,10)      

           outputvidep.append(frame)
       return outputvidep   


    






