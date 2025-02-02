from  ultralytics import YOLO  
import cv2
import supervision as sv
import pickle
from utilis.utilis import read
import numpy as np
from sklearn.cluster import KMeans

class Tracker:
    def __init__(self,model_p):
        self.model=YOLO(model_p)
        self.tracker=sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub:
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

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
    def draw_ellipse(self,frame,bbox,color,track_id):
        y2=int(bbox[3])
        x_center=int((bbox[0]+bbox[2])/2)
        y_center=int((bbox[1]+bbox[3])/2)
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
        text = str(track_id)
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
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

        return frame

    def draw_traingle(self,frame,bbx,color):
        x_center=int((bbx[0]+bbx[2])/2)
        y_center=int((bbx[1]+bbx[3])/2)
        triangle_points = np.array([[x_center-10,y_center-10], [x_center,y_center], [x_center+10, y_center-10]], np.int32) 
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
       video_frames=read(video)
       players_dict=tracks['players'] 
       referers_dict=tracks['referees']
       balls=tracks['ball'] 
       outputvidep=[]
       for frame_num,frame in enumerate(video_frames):  
           frame=frame.copy()
           for track_id,player in players_dict[frame_num].items():
               dominant_clr=self.get_dominant_color(np.array(frame),player["bbox"])
               frame=self.draw_ellipse(frame,player["bbox"],dominant_clr,track_id)  
           for track_id,referee in referers_dict[frame_num].items():
               frame=self.draw_ellipse(frame,referee["bbox"],(0,255,255),track_id)
           for track_id,ball in balls[frame_num].items():
               frame=self.draw_traingle(frame,ball['bbox'],(0,255,0))      

           outputvidep.append(frame)
       return outputvidep   


    






