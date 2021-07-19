from PIL import Image
import numpy as np
import os


root = ".\\rgbd-scenes-v2\\train\\"

for path, subdirs, files in os.walk(root):
    
    for fname in files:
        f = re.split('[.\-]', fname) 

        assert(len(f) == 3)

        instance = f[0]
        img_type = f[1]
        ext = f[2]

        if img_type == "depth":
            img_path = os.path.join(path, fname)
            img=Image.open(img_path)
            img=np.array(img)

            out_path = os.path.join(path, instance+"-truemask."+ext)

            im = Image.fromarray(img==0.0)
            im = im.convert('1') # convert image to black and white
            print(out_path)
            im.save(out_path)
