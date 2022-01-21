import os
import cv2
from tqdm import tqdm

anno_file = '/media/aiteam/storage/AI_DATA/fire_smoke_dataset/eda/mAP/test_new_label/data/val.txt'
dst_file = '/home/primedo/ai_server/fire_detection/yolov4-tiny-pytorch/model_data/val.txt'
# img_folder  = '/media/aiteam/storage/AI_DATA/fire_smoke_dataset/label_data/ip_data_ver1/images'
with open(dst_file,'w') as f:
    txt = ''
    with open(anno_file, 'r') as f1:
        datas = f1.read().splitlines()  
        for data in datas:
            image_path = data
            txt += image_path
            prefix = os.path.basename(image_path).split('.')[-1]
            h,w = cv2.imread(image_path).shape[:2]
            txt_file = image_path.replace(prefix,'txt')
            print(txt_file)
            with open(txt_file, 'r') as f2:
                labels = f2.read().splitlines()
                for label in labels:
                    label = label.split(' ')
                    x_center, y_center = float(label[1]), float(label[2])
                    width, height = float(label[3]), float(label[4])
                    obj_class = str(int(label[0]))
                    xmin = str(int((x_center - width/2)*w))
                    xmax = str(int((x_center + width/2)*w))
                    ymin = str(int((y_center - height/2)*h))
                    ymax = str(int((y_center + height/2)*h))
                    txt += ' ' + ','.join([xmin,ymin,xmax,ymax,obj_class])
            txt += '\n'
            #print(txt)
        f.write(txt)
