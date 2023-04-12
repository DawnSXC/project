

from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib.pyplot as plt
cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [3.12493437e-02, 2.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 3.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 4.00000000e+00, 1.31250131e-06],
                     [3.12493437e-02, 5.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 6.00000000e+00, 1.31250131e-06],
                     [3.12493437e-02, 7.00000000e+00, 9.37500000e-02]
                 ])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True


cv2.namedWindow('show3d', cv2.WINDOW_NORMAL)
cv2.moveWindow('show3d',400, 95)
cv2.setMouseCallback('show3d', onmouse)

dll = ct.cdll.LoadLibrary('./render_balls.dll')


def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=True, magnifyBlue=0, freezerot=False,
               background=(0,0,0), normalizecolor=True, ballradius=7,size=100):
    global showsz, mousex, mousey, zoom, changed

    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz

    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 255
        c1 = np.zeros((len(xyz),), dtype='float32') + 255
        c2 = np.zeros((len(xyz),), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]




    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        rotmat = np.eye(3)
        if not freezerot:
            xangle = (mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)), (30, showsz - 30), 0, 0.5,
                        (255,255,255))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)), (30, showsz - 50), 0, 0.5,
                        (255,255,255))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0, 0.5, (255,255,255))

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q') or size == 99:
            print(cmd)#113
            size =100
            break
        elif cmd == ord('Q') or size == 99:
            size=100
            sys.exit(0)

        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    print(c0)
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0

            c0 = np.require(c0, 'float32', 'C')
            c1 = np.require(c1, 'float32', 'C')
            c2 = np.require(c2, 'float32', 'C')
            changed = True

        if cmd == ord('n') or size==110:
            zoom *= 1.1
            changed = True
            size=100
        elif cmd == ord('m') or size == 109:
            print(cmd)#109
            zoom /= 1.1
            changed = True
            size=100
        elif cmd == ord('r') or size == 114:
            print(cmd)
            zoom = 1.0
            changed = True
            size=100
        elif cmd == ord('s') or size == 115:
            print(cmd)
            cv2.imwrite('show3d.png', show)
            size=100
        if waittime != 0:
            break
    return cmd


def pc_normalize(pc):  #点云数据归一化
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet_cls2', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()
#加载数据集





def test(model,point_set, num_class=40, vote_num=1,flag=1,size=100,color=0,p_size=7):
    #color=int(color*(9/255))
    #print(color)
    #mean_correct = []
    n_points = point_set
    point_set = torch.as_tensor(point_set)
    point_set = point_set.to(device)
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    vote_pool = torch.zeros(1, 40).to(device)
    for _ in range(vote_num):
        pred, _,x2 = classifier(point_set)
        x2=x2.cpu().detach().numpy()
        x2=np.squeeze(x2)
        x2=np.unique(x2)
        vote_pool += pred
    pred = vote_pool / vote_num
    # 对预测结果每行取最大值得到分类
    pred_choice = pred.data.max(1)[1]
    #print(pred_choice.shape)
    #可视化
    #print(x2.shape)
    #print(type(x2))
    file_dir = 'visualize_original/'
    save_name_prefix = 'pred'
    #draw(n_points[:, 0, x2], n_points[:, 1, x2], n_points[:, 2, x2], save_name_prefix, file_dir, color=pred_choice)
    if flag==1:
        a=np.concatenate((n_points[:, 0, x2], n_points[:, 1, x2], n_points[:, 2, x2]),axis=0)
        b=a.T
        seg=np.full((int(b.size / 3)), color)

        #print(np.full((int(b.size/3)),2).shape)
        #print((np.full((int(b.size/3)),2))[:])
        cmap = plt.cm.get_cmap("hsv", 10)
        cmap = np.array([cmap(i) for i in range(10)])[:, :]
        gt = cmap[seg-1, :]
        pred_color=cmap[np.full((int(b.size/3)),color)[:],:]
        showpoints(b,size=size,c_gt=gt,c_pred=pred_color,ballradius=p_size)
    else:
        a = np.concatenate((n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :]), axis=0)
        b = a.T
        seg = np.full((int(b.size / 3)), color)
        # print(np.full((int(b.size/3)),1).shape)
        # print((np.full((int(b.size/3)),1))[:])
        cmap = plt.cm.get_cmap("hsv", 10)
        cmap = np.array([cmap(i) for i in range(10)])[:, :]
        gt = cmap[seg-1, :]
        pred_color=cmap[(np.full((int(b.size/3)),color))[:], : ]
        showpoints(b, size=size,c_gt=gt,c_pred=pred_color,ballradius=p_size)

    return pred_choice


def view(class_name,index,flag,size=100,color=0,p_size=7):
    dataset = 'data/modelnet40_normal_resampled/'+class_name+'/'+class_name+'_'+index+'.txt'
    pcdataset = np.loadtxt(dataset, delimiter=',').astype(np.float32)
    point_set = pcdataset[0:1024, :]
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    point_set = point_set[:, 0:3]
    point_set = point_set.transpose(1, 0)
    # print(point_set.shape)
    point_set = point_set.reshape(1, 3, 1024)
    def log_string(str):
        logger.info(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    '''CREATE DIR'''
    experiment_dir = 'E:\Pointnet_Pointnet2_pytorch/log/classification/pointnet_cls2'
    #print(experiment_dir)
    '''LOG'''
    #args = parse_args()
    logger = logging.getLogger("Model")
    '''
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    '''
    num_class = 40
    #选择模型
    #model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    #print(model_name)
    model_name='pointnet_cls'
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=False)
    #if not args.use_cpu:
    classifier = classifier.to(device)
    #选择训练好的.pth文件
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    #预测分类
    print('color:')
    print(color)
    with torch.no_grad():
         pred_choice = test(classifier.eval(), point_set, vote_num=3, num_class=num_class,flag=flag,size=size,color=color,p_size=p_size)
         #log_string('pred_choice: %f' % (pred_choice))

