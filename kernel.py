
from collections import defaultdict
import csv
import sys

import copy
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import math
import pandas as pd
import matplotlib.pyplot
import random
import mxnet as mx
from datetime import datetime



csv.field_size_limit(sys.maxsize);

DF = pd.read_csv('data/train_wkt_v4.csv')
ids = sorted(DF.ImageId.unique())

poly_types=range(1,11)

def addtwodimdict(thedict, key_a, key_b, val): 
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})

x_max={}
y_min={}
train_polygons={}
im_rgb={}
im_size={}

for IM_ID in ids:
    for POLY_TYPE in poly_types:
        # Load grid size
        for _im_id, _x, _y in csv.reader(open('data/grid_sizes.csv')):
            if _im_id == IM_ID:
                addtwodimdict(x_max, IM_ID, POLY_TYPE, float(_x))
                addtwodimdict(y_min, IM_ID, POLY_TYPE, float(_y))
                break

        # Load train poly with shapely
        for _im_id, _poly_type, _poly in csv.reader(open('data/train_wkt_v4.csv')):
            if _im_id == IM_ID and _poly_type == str(POLY_TYPE):
                addtwodimdict(train_polygons, IM_ID, POLY_TYPE, shapely.wkt.loads(_poly))
                break

        # Read image with tiff
        addtwodimdict(im_rgb, IM_ID, POLY_TYPE, tiff.imread('data/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0]))
        addtwodimdict(im_size, IM_ID, POLY_TYPE, im_rgb[IM_ID][POLY_TYPE].shape[:2])
train_polygons_scaled={}

def get_scalers(x_max,y_min,im_size):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (float(w) / (w + 1))
    h_ = h * (float(h) / (h + 1))
    return w_ / x_max, h_ / y_min

for IM_ID in ids:
    for POLY_TYPE in poly_types:
        x_scaler, y_scaler = get_scalers(x_max[IM_ID][POLY_TYPE],y_min[IM_ID][POLY_TYPE],im_size[IM_ID][POLY_TYPE])
        addtwodimdict(train_polygons_scaled, IM_ID, POLY_TYPE, shapely.affinity.scale(train_polygons[IM_ID][POLY_TYPE], xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)))
train_mask={}

def mask_for_polygons(polygons,im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

for IM_ID in ids:
    for POLY_TYPE in poly_types:
        addtwodimdict(train_mask, IM_ID, POLY_TYPE, mask_for_polygons(train_polygons_scaled[IM_ID][POLY_TYPE],im_size[IM_ID][POLY_TYPE]))
# rotate(): rotate image  
# return: rotated image object  
def rotate(  
    img,  #image matrix  
    angle #angle of rotation  
    ):  
      
    height = img.shape[0]  
    width = img.shape[1]  
      
    if angle%180 == 0:  
        scale = 1  
    elif angle%90 == 0:  
        scale = float(max(height, width))/min(height, width)  
    else:  
        scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)  
      
    #print 'scale %f\n' %scale  
          
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)  
    rotateImg = cv2.warpAffine(img, rotateMat, (int(width/scale), int(height/scale)) ) 
    #cv2.imshow('rotateImg',rotateImg)  
    #cv2.waitKey(0)  
      
    return rotateImg #rotated image 

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

class SimpleIter:
    def __init__(self, num_batches=1280000000000000000):
        self.num_batches = num_batches
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0        

    def __next__(self):
        return self.next()
    
    @property
    def provide_data(self):
        return zip(['data'],[(128,3,64,64)])

    @property
    def provide_label(self):
        return zip(['softmax_label'],[(128,256)])
    
    def next(self):
        temp_rgb={}
        temp_mask={}
        if random.randint(0,3)!=0:
            for anid in ids:
                temp_ang=random.randint(0,359)
                temp_rgb[anid]=rotate(im_rgb[anid][1],temp_ang)
                for m in range(1,11):
                    addtwodimdict(temp_mask, anid , m, rotate(train_mask[anid][m],temp_ang))
        else:
            for anid in ids:
                temp_rgb[anid]=im_rgb[anid][1]
                for m in range(1,11):
                    addtwodimdict(temp_mask, anid , m, train_mask[anid][m])
        if self.cur_batch < self.num_batches:
            data=[]
            label=[]
            self.cur_batch += 1
            for batch_num in range(0,12800):   
                im_id=random.sample(ids,1)[0]
                x_samp=random.randint(0,temp_rgb[im_id].shape[0]-64)
                y_samp=random.randint(0,temp_rgb[im_id].shape[1]-64)
                temp_data = temp_rgb[im_id][:][x_samp:x_samp+64,y_samp:y_samp+64]
                temp_data = temp_data.transpose([2, 0, 1])
                data.append(temp_data)
                temp_label=[]
                for i in range(0,16):
                    for j in range(0,16):
                        bit=0
                        for area_type in range(1,11):
                            if int(temp_mask[im_id][area_type][x_samp+24+i][y_samp+24+j])==1:
                                bit=area_type
                        temp_label.append(bit)
                label.append(temp_label)
            return SimpleBatch(data, label)
        else:
            raise StopIteration
smpiter=SimpleIter()
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
soft_name=[]
for l in range(0,256):
    soft_name.append('_'+str(l)+'_softmax_label')
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act


net = mx.sym.Variable("data")
net = ConvFactory(data=net, num_filter=64, kernel=(16,16), stride=(4,4), name='conv_1')
net = mx.sym.Pooling(data=net, pool_type="max", kernel=(2,2), stride=(1,1))
net = ConvFactory(data=net, num_filter=112, kernel=(4,4), stride=(1,1), name='conv_2')
net = ConvFactory(data=net, num_filter=80, kernel=(3,3), stride=(1,1), name='conv_3')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=4096)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
# net = mx.sym.FullyConnected(data=net, num_hidden=11)
# net = mx.sym.SoftmaxOutput(data=net, name='softmax', multi_output=True)
soft_lay=[]
for j in range(0,256):
    temp_net = mx.sym.FullyConnected(data=net, num_hidden=11)
    soft_lay.append(mx.sym.SoftmaxOutput(data=temp_net, name='_'+str(j)+'_softmax'))
group = mx.symbol.Group(soft_lay)
# shape = {"data" : (128, 3, 64, 64)}
# mx.viz.plot_network(symbol=soft_lay[0], shape=shape)
#group.list_outputs()
mod = mx.mod.Module(symbol=group, 
                    context=mx.gpu(),
                    data_names=["data"], 
                    label_names=soft_name)
for lops in range(0,1):
    test=smpiter.next()
    dt=mx.nd.array(test.data)
    temp_lbl =(np.array(test.label).transpose([1, 0]))
    lbl=[]
    for z in range(0,256):
        lbl.append(mx.nd.array(temp_lbl[z]))
    train_iter = mx.io.NDArrayIter(dt, lbl, 128, shuffle=True)
    a=datetime.now()
    mod.fit(train_iter, 
            eval_data=train_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate':0.0005,'momentum':0.9,'wd':0.0002},
            eval_metric='acc',
            num_epoch=1)
    print (datetime.now()-a)
mod.load_params('1000epo_rotate')
for lops in range(0,1):
    test=smpiter.next()
    dt=mx.nd.array(test.data)
    temp_lbl =(np.array(test.label).transpose([1, 0]))
    lbl=[]
    for z in range(0,256):
        lbl.append(mx.nd.array(temp_lbl[z]))
    train_iter = mx.io.NDArrayIter(dt, lbl, 128, shuffle=True)
    print mod.score(train_iter, eval_metric='acc')
def pred_a_window(img):
    pred_iter = mx.io.NDArrayIter(mx.nd.array(img),batch_size=128)
    nouselab=mod.predict(pred_iter)
    output=[]
    for bits in nouselab:
        output.append(bits.asnumpy())
    return np.array(output).transpose([1,2,0])
def pred_with_mod(img):
    pred_list=[]
    pred_mask=[]
    for i in range(0,10):
        pred_mask.append(np.zeros(img.shape[1:]))
    for xs in range(0,img.shape[1]-64,16):
        for ys in range(0,img.shape[2]-64,16):
            pred_list.append(img[:,xs:xs+64,ys:ys+64])
        pred_list.append(img[:,xs:xs+64,img.shape[2]-64:img.shape[2]])
    xs=img.shape[1]-64
    for ys in range(0,img.shape[2]-64,16):
        pred_list.append(img[:,xs:xs+64,ys:ys+64])
    pred_list.append(img[:,xs:xs+64,img.shape[2]-64:img.shape[2]])
    output_list=pred_a_window(pred_list)
    
    count_list=0
    for xs in range(0,img.shape[1]-64,16):
        for ys in range(0,img.shape[2]-64,16):
            for i in range(0,10):
                pred_mask[i][xs+24:xs+40,ys+24:ys+40]=output_list[count_list][i+1].reshape(16,16)
            count_list += 1
        for i in range(0,10):
            pred_mask[i][xs+24:xs+40,img.shape[2]-40:img.shape[2]-24]=output_list[count_list][i+1].reshape(16,16)
        count_list += 1
    xs=img.shape[1]-64
    for ys in range(0,img.shape[2]-64,16):
        for i in range(0,10):
            pred_mask[i][xs+24:xs+40,ys+24:ys+40]=output_list[count_list][i+1].reshape(16,16)
        count_list += 1
    for i in range(0,10):
        pred_mask[i][xs+24:xs+40,img.shape[2]-40:img.shape[2]-24]=output_list[count_list][i+1].reshape(16,16)
    count_list += 1
    return pred_mask
def mask_to_polygons(mask, epsilon=5., min_area=1.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

df = pd.read_csv('data/sample_submission.csv')
GS = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
idss = sorted(df.ImageId.unique())
threshold = 0.1
temp_df=pd.DataFrame()
for oneid in idss:
    print oneid
    temp_df=pd.DataFrame()
    temp_im=tiff.imread('data/three_band/{}.tif'.format(oneid))
    temp_mask=pred_with_mod(temp_im)
    pred_binary_mask = np.array(temp_mask) >= threshold
    for i in range(0,10):
        j=i+1
        pred_polygons = mask_to_polygons(pred_binary_mask[i])
        x_max = GS.loc[GS['ImageId'] == oneid, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == oneid, 'Ymin'].as_matrix()[0]
        x_scaler, y_scaler = get_scalers(x_max, y_min, pred_binary_mask[i].shape)
        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
        temp_df=temp_df.append(pd.DataFrame([[oneid,j,scaled_pred_polygons]],columns=['ImageId','ClassType','MultipolygonWKT']))
temp_df.to_csv('sub.csv', index=False)
