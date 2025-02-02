import cv2
 
def  read(video_path):
    capture=cv2.VideoCapture(video_path)
    video_frames=[]
    while True:
        isThere,frame=capture.read()
        if not isThere:
            break
        video_frames.append(frame)

    return video_frames

def output(video_frames,fps,output_path=None):
    if output_path ==None:
        output_path=="output_video.avi"
    fourc=cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter(output_path,fourc,fps,(video_frames[0].shape[1],video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame)
    out.release()    
    

