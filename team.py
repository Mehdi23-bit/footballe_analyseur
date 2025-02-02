import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
class Teams:
    def __init__(self):
        self.teams={}
        self.players={}
    def resize(image):
        img=image
        height,width,something_i_dont_what_it_is=img.shape

        uperhalf=img[:int(height/2),:]
        bigger_image = cv2.resize(uperhalf, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
        return bigger_image

    def get_player_color(self,frame):
        image=frame
        img=self.resize(image)
        pixels = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2, random_state=0)  
        kmeans.fit(pixels)
        labels=kmeans.labels_
        clustered_image=labels.reshape(img.shape[0],img.shape[1])
        corner=[clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        nplayer=max(set(corner),key=corner.count)
        print(nplayer)
        player_cluster=1-nplayer 
        print(kmeans.cluster_centers_[player_cluster])
        return kmeans.cluster_centers_[player_cluster]
        
    def get_teams_color(self,tracks,frame):
        colors=[]
        for id,player in tracks['players'][0].items():
            bbox=tracks['players'][0][id]['bbox']
            image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            colors.append(self.get_player_color(image))
        kmean=KMeans(n_clusters=2, random_state=0)
        kmean.fit(colors)  
        self.kmean=kmean
        self.team[0]=kmean.cluster_centers_[0]
        self.team[1]=kmean.cluster_centers_[1] 
        
    def set_players(self,tracks,frame):
      for id,player in tracks['players'][0].items():
         bbox=tracks['players'][0][id]['bbox']
         image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
         color=self.get_player_color(image)
         team_id=self.kmean.predict(color.reshape(1,-1))[0]
         self.players[id]=self.teams[team_id]


    
    