import shutil

import os

folder_path = "C:/Users/elmeh/Desktop"
contents = os.listdir(folder_path)
for cont in contents:
   if cont[-4:]!='.lnk':
      shutil.move(f"{folder_path}/{cont}",f"{folder_path}/Desktop/")
