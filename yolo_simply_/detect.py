import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from model.net import Yolo_V2
import os
import utils

class Detect:

    def __init__(self,img):
        # 初始化参数
        self.iou_thresh = 0.3
        self.net = Yolo_V2().eval()    #测试
        # self.net_param = r"./param/yolo_v2_test_100.pkl"      #过拟合模型
        self.net_param = r"/home/tensorflow01/桌面/yolo_simply_/param/yolo_v2_params.pkl"          #重新训练模型
        self.img = Variable(torch.Tensor(img))
        # cuda
        # if torch.cuda.is_available():
        #     self.net = self.net.cuda()
        #     self.img = self.img.cuda()
        # 加载网络参数
        self.net.load_state_dict(torch.load(self.net_param))

    def detect(self):
        out_cond,out_offset = self.net(self.img)
        out_cond = out_cond.view(9, 7, 7,1)                      #维度变形  view
        out_offset = out_offset.view(-1 ,9, 4, 7, 7)
        out_offset = out_offset.permute(0,1,3,4,2)
        out_offset = out_offset.cpu().data.numpy()

        """因为每个CELL只预测一个框，所以选出置信度最大的channl"""
        out_cond_max,idx = torch.max(out_cond,dim=0)
        print(out_cond_max.shape)
        # exit()
        "取出每个CELL对应通道的box，共49个框"
        idx = idx.view(7*7)
        boxes = []
        i = 0
        for h in range(7):
            for w in range(7):
                box = out_offset[0][idx[i]][h][w][:]
                box = np.insert(box,4,idx[i])
                boxes.append(box)
                i += 1
        # print(boxes)
        boxes = np.reshape(boxes,newshape=(7,7,5))
        # exit()
        boxes = torch.Tensor(boxes)
        print(boxes.shape)
        # exit()
        """置信度删选，得到cond>0.97的索引"""
        # for j in range(len(idx))
        index_cond = torch.gt(out_cond_max,0.98)
        # print(_boxes.shape)
        # exit()
        index_nonzero = torch.nonzero(index_cond)
        # print(index_nonzero)
        # exit()
        index_nonzero_np = index_nonzero.cpu().data.numpy()
        boxes = boxes.cpu().data.numpy()
        _boxes = []
        for i in range(len(index_nonzero_np)):
            h,w = index_nonzero_np[i][0],index_nonzero_np[i][1]
            # print(c,"--",h,"--",w)
            # print(out_offset_np[0][c][h][w][0])
            # print(out_offset_np[0][c][h][w][1])
            # print(out_offset_np[0][c][h][w][2])
            # print(out_offset_np[0][c][h][w][3])
            # print(out_cond_np[0][c][h][w][0])
            # exit()
            [x1,y1,x2,y2,cons] = self.reverse_boxes(boxes[h][w][4],h,w,boxes[h][w][0],boxes[h][w][1],
                                                    boxes[h][w][2],boxes[h][w][3],
                                                    out_cond_max[h][w][0])
            _boxes.append([x1,y1,x2,y2,cons])
        if len(_boxes) == 0:
            print("没有目标物"),
            return np.array([])                     #没有框返回空array
        _boxes = np.stack(_boxes)
        # print(boxes.shape)
        # 做NMS删选
        _boxes = utils.nms(_boxes, self.iou_thresh, isMin=False)
        # print("nms后的BOX==>", boxes.shape)
        # exit()
        return _boxes

    def reverse_boxes(self,c,h,w,offset_x1,offset_y1,offset_x2,offset_y2,cons):

        cx,cy = w*32+16,h*32+16
        offset = [offset_x1,offset_y1,offset_x2,offset_y2]
        # 反算偏移值
        if c == 0:
            box = [cx - 16, cy -16, cx + 16, cy + 16]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 1:
            box = [cx - 24, cy - 24, cx + 24, cy + 24]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 2:
            box = [cx - 32, cy - 32, cx + 32, cy + 32]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 3:
            box = [cx - 24, cy - 8, cx + 24, cy + 8]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 4:
            box = [cx - 48, cy - 16, cx + 28, cy + 16]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 5:
            box = [cx - 96, cy - 32, cx + 96, cy + 32]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 6:
            box = [cx - 8, cy - 24, cx + 8, cy + 24]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 7:
            box = [cx - 16, cy - 48, cx + 16, cy + 48]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 8:
            box = [cx - 32, cy - 96, cx + 32, cy + 96]
            x1, y1, x2, y2 = self.get_box_value(box, offset)

        return [x1,y1,x2,y2,cons]

    def get_box_value(self,box,offset):
        x1 = int(offset[0] * (box[2] - box[0]) + box[0])
        y1 = int(offset[1] * (box[3] - box[1]) + box[1])
        x2 = int(offset[2] * (box[2] - box[0]) + box[2])
        y2 = int(offset[3] * (box[3] - box[1]) + box[3])
        return x1,y1,x2,y2

if __name__ == '__main__':
    ttfont = ImageFont.truetype(font="/home/tensorflow01/oneday/yolo_v1自我实现/NotoSansCJK-Black.ttc", size=20)
    test_img_dir = r"/home/tensorflow01/桌面/yolo_simply_/test_image"
    list = os.listdir(test_img_dir)
    for name in list:
        test_img = os.path.join(test_img_dir,name)
        img = Image.open(test_img)
        img = img.resize((224,224))   #不是224*224，resize 成224
        img_in = np.array(img, dtype=np.float32)/255.0 - 0.5
        img_in = np.transpose(img_in, (2, 0, 1))
        img_in = np.array([img_in])

        detect = Detect(img_in)
        boxes = detect.detect()
        imDraw = ImageDraw.Draw(img)
        # boxes = [[10, 20, 129, 255],[126, 0, 221, 164]]
        for i in range(boxes.shape[0]):
            print(boxes[i])
            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])
            imDraw.rectangle((x1, y1, x2, y2), outline="red")
        img.save(r"/home/tensorflow01/桌面/yolo_simply_/test_img_result/{}".format(name))
        # img.save(r"/home/tensorflow01/桌面/yolo_simply_/test_img_100_params_result/{}".format(name))
        img.show()





