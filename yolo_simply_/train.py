import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os,shutil
from data.sampling import MyDataset
from model.net import Yolo_V2

if __name__ == '__main__':

    yolo_v2 = Yolo_V2().train()  # 创建网络
    label_txt = r"/media/tensorflow01/myfile/0_yolo_VOC/2007_train/2012+2007_train_label.txt"  #17123+5010
    mydataset = MyDataset(label_txt)
    dataloader = data.DataLoader(mydataset, batch_size=64, shuffle=True)

    if torch.cuda.is_available():
        yolo_v2 = yolo_v2.cuda()

    # 加载保存的数据
    # yolo_v2_paramater_path = r"/home/tensorflow01/桌面/yolo_simply_param/yolo_v2_test_100.pkl"#加载过拟合模型（１００测试模型）
    yolo_v2_paramater_path = r"/home/tensorflow01/桌面/yolo_simply_/param/yolo_v2_params.pkl"  #重新训练

    if os.path.exists(yolo_v2_paramater_path):
        yolo_v2.load_state_dict(torch.load(yolo_v2_paramater_path))
        print("module restore")
    if not os.path .exists(yolo_v2_paramater_path): #创建参数文件
        dirs,file = os.path.split(yolo_v2_paramater_path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    # 定义优化器
    opt_yolo_v2 = optim.Adam(yolo_v2.parameters(),lr=0.00001,weight_decay=0.0001)

    # 定义损失函数
    loss_cond_fun = nn.MSELoss()
    loss_offset_fun = nn.MSELoss()

    for i in range(10000):
        img,cond,offset = mydataset.get_batch(dataloader)
        # img, cond, offset = Variable(img),Variable(cond),Variable(offset)
        if torch.cuda.is_available():
            img, cond, offset = img.cuda(),cond.cuda(),offset.cuda()
        # 将参数传入网络
        out_cond,out_offset = yolo_v2(img)
        out_cond = out_cond.view(-1, 9, 7, 7, 1)     # 维度变形  view
        out_offset = out_offset.view(-1, 9, 4, 7, 7)
        out_offset = out_offset.permute(0,1,3,4,2)

        #取出对应的索引计算损失
        idx1 = torch.ne(cond[:,:,:,:,0],0)
        cond_1 = cond[idx1].double()
        # print(cond_1.shape)
        offset_1 = offset[idx1]

        idx0 = torch.eq(cond[:,:,:,:,0],0)
        cond_0 = cond[idx0].double()

        _1_out_cond = out_cond[idx1].double()
        _0_out_cond = out_cond[idx0].double()
        # print(_1_out_cond.shape)
        # exit()
        _1_out_offset = out_offset[idx1].double()

        loss_cond = loss_cond_fun(_1_out_cond,cond_1) + loss_cond_fun(_0_out_cond, cond_0)
        loss_offset = loss_offset_fun(_1_out_offset, offset_1)
        loss = loss_cond + loss_offset*5

        opt_yolo_v2.zero_grad()   # 优化器清零
        loss.backward()           # 损失反向
        opt_yolo_v2.step()        # 更新优化器

        print(i,"-- ",loss.cpu().data.numpy(),"--loss_cond:",loss_cond.cpu().data.numpy()," --loss_offset:",loss_offset.cpu().data.numpy())

        if ((i + 1) % 100 == 0):
            torch.save(yolo_v2.state_dict(), yolo_v2_paramater_path)
            print("module save")

        if((i+1) % 500 == 0): #  复制参数以防参数文件训练过程中损坏
            shutil.copyfile(yolo_v2_paramater_path,os.path.join(r"/home/tensorflow01/桌面/yolo_simply_/param","copy_yolo_v2_params.pkl"))# DST必须是完整的目标文件名
            # shutil.copyfile(yolo_v2_paramater_path,os.path.join(r"/home/tensorflow01/桌面/yolo_simply_/param","copy_yolo_v2_params.pkl"))




