import os
import re

def generate_set(root, color_file, mask_file):
    color_img = []
    mask_img = []

    for path, subdirs, files in os.walk(root):
        for fname in files:
            print(fname)
            f = re.split('[.\-]', fname) 

            assert(len(f) == 3)

            instance = f[0]
            img_type = f[1]
            ext = f[2]
            #only getting 100 instances for each scene
            if int(instance)<100:
                img_path = os.path.join(path, fname)
                if img_type == "color":
                    color_img.append(img_path)

                if img_type == "truemask":
                    mask_img.append(img_path)

    assert( len(color_img) == len(mask_img) )

    for i in range(len(color_img)):
        print( color_img[i] , mask_img[i] )
        assert( color_img[i][:36] == mask_img[i][:36] )


    with open(color_file, 'w') as filehandle:
        filehandle.write('%s\n' % 'img')
        for item in color_img:
            filehandle.write('%s\n' % item)

    with open(mask_file, 'w') as filehandle:
        for item in mask_img:
            filehandle.write('%s\n' % item)


test_root = ".\\rgbd-scenes-v2\\test\\"
test_color_file = "test_color_img.csv"
test_mask_file = "test_mask_img.csv"
generate_set(test_root, test_color_file, test_mask_file)

train_root = ".\\rgbd-scenes-v2\\train\\"
train_color_file = "train_color_img.csv"
train_mask_file = "train_mask_img.csv"
generate_set(train_root, train_color_file, train_mask_file)