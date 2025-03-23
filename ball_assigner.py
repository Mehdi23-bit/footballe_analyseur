import sys 
sys.path.append('../')


class PlayerBallAssigner():
    def get_center_of_bbox(bbox):
        x1,y1,x2,y2 = bbox
        return int((x1+x2)/2),int((y1+y2)/2)

    def get_bbox_width(bbox):
        return bbox[2]-bbox[0]

    def measure_distance(p1,p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    def measure_xy_distance(p1,p2):
        return p1[0]-p2[0],p1[1]-p2[1]

    def get_foot_position(bbox):
        x1,y1,x2,y2 = bbox
        return int((x1+x2)/2),int(y2)
    def __init__(self):
         self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = self.get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = self.measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = self.measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player