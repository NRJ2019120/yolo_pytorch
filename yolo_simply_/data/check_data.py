import PIL.Image as Image
from PIL import ImageDraw
from PIL import ImageFont

"""check data lebal"""

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

def index_to_cissify_name(index):
    num = int(index)
    classify_name = VOC_CLASSES[num]
    return classify_name

def get_box_value(box,offset):

    x1 = int(offset[0] * (box[2] - box[0]) + box[0])
    y1 = int(offset[1] * (box[3] - box[1]) + box[1])
    x2 = int(offset[2] * (box[2] - box[0]) + box[2])
    y2 = int(offset[3] * (box[3] - box[1]) + box[3])
    return x1,y1,x2,y2

def reverse_boxes(c, h, w, offset_x1, offset_y1, offset_x2, offset_y2):
    cx, cy = w * 32 + 16, h * 32 + 16
    offset = [offset_x1, offset_y1, offset_x2, offset_y2]
    # 反算偏移值
    if c == 0:
        box = [cx - 16, cy - 16, cx + 16, cy + 16]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 1:
        box = [cx - 24, cy - 24, cx + 24, cy + 24]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 2:
        box = [cx - 32, cy - 32, cx + 32, cy + 32]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 3:
        box = [cx - 24, cy - 8, cx + 24, cy + 8]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 4:
        box = [cx - 48, cy - 16, cx + 28, cy + 16]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 5:
        box = [cx - 96, cy - 32, cx + 96, cy + 32]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 6:
        box = [cx - 8, cy - 24, cx + 8, cy + 24]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 7:
        box = [cx - 16, cy - 48, cx + 16, cy + 48]
        x1, y1, x2, y2 = get_box_value(box, offset)
    if c == 8:
        box = [cx - 32, cy - 96, cx + 32, cy + 96]
        x1, y1, x2, y2 = get_box_value(box, offset)

    return [x1, y1, x2, y2]
if __name__ == '__main__':

    ttfont = ImageFont.truetype(font="/home/tensorflow01/oneday/yolo_v1自我实现/NotoSansCJK-Black.ttc", size=20)
    xml_path = r"/media/tensorflow01/myfile/0_yolo_VOC/2012_train/Annotations"
    img_path = r"/media/tensorflow01/myfile/0_yolo_VOC/2012_train/JPEGImages"
    save_img_224_path = r"/media/tensorflow01/myfile/0_yolo_VOC/2012_train/224"

    # save_txt = r"/media/tensorflow01/myfile/0_yolo_VOC/2012_train/label.txt"
    # save_txt = r"/media/tensorflow01/myfile/0_yolo_VOC/2012_test/2012_all_label.txt"  # 2012 train + test lebal+5138
    """测试集标签不太好"""
    save_txt = r"/media/tensorflow01/myfile/0_yolo_VOC/2007_train/2012+2007_train_label.txt"  # 2012 train + 2007train+5010
    lines = open(save_txt).readlines()
    # print(len(lines))  #22261=17123+5138
    # exit()             #22134 = 17123+5010
    for i in range(17123+490,17123+500):    #此处注意 不同lebal，数量不同！！
        strs = lines[i].split("*")
        print(strs)
        print(len(strs))
        im_path = strs[0]
        img = Image.open(im_path)
        imdraw = ImageDraw.Draw(img)
        for i in range(1,len(strs)):
            list = strs[i].split(",")
            print(list)
            off_x1 = float(list[-4])
            off_y1 = float(list[-3])
            off_x2 = float(list[-2])
            off_y2 = float(list[-1])
            cls = list[0]
            w_id = float(list[2])
            h_id = float(list[1])
            channl = float(list[3])
            # 反算坐标
            box = reverse_boxes(channl,h_id,w_id,off_x1,off_y1,off_x2,off_y2)
            # cls_name =index_to_cissify_name(cls)
            imdraw.rectangle(box, outline="red", width=5)
            imdraw.text((box[0], box[1]), cls , fill=(0, 255, 0), font=ttfont)
        print("=========")
        img.show()
