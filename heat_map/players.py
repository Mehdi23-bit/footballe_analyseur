import pickle


with open("/home/elmehdi/Desktop/footballe_analyseur/ball_positions.pkl","rb")as f:
    positions=pickle.load(f)


print(positions)    

    