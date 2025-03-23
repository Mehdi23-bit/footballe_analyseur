from  ultralytics import YOLO
import pickle
import supervision as sv
class PitchTracker:
    def __init__(self,model):
        self.model=YOLO(model)
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
            "pitch":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

           
            detection_supervision = sv.Detections.from_ultralytics(detection)

            
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["pitch"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['pitch']:
                    tracks["pitch"][frame_num][track_id] = {"bbox":bbox}
                         
            

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return cls_names_inv    