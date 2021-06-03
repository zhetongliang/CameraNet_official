import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import csv,cv2,glob
import tensorflow as tf
import sys
import tensorflow as tf
import layers,pre_process
import scipy.io as scio 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.contrib.slim as slim
#from justin_loss import gradient_loss,cosine_dist
import datapipeline as dp
import loss_func as lf
import pre_process as pp
import numpy as np
import os,time,cv2,math,rawpy,sys,my_model_pool


base = './Data'
test_input_path = base+'/SID/input'
test_middle_path = base+'/SID/middle'
test_output_path = base+'/SID/output'
input_path = test_input_path
middle_path = test_middle_path
output_path = test_output_path
test_input_filelist = base+'/SID/inputlist_test.txt'
test_output_filelist = base+'/SID/outputlist_test.txt'
ckpt = '/Ckpt/CameraNet_SID'
test_epoch = 4800
stage = 'all'
ckpt_model_path = ckpt + '/%s_%04d/model.ckpt'%(stage,test_epoch)

with open(test_input_filelist, 'r') as fid:
    input_filelist = [l.strip() for l in fid.readlines()]
with open(test_output_filelist, 'r') as fid:
    output_filelist = [l.strip() for l in fid.readlines()]

def clipping(x):
   return tf.minimum(tf.maximum(x,0.0),1.0)

def RestoreNet(input,reuse=False):
    ## Restore-Net 
    with tf.variable_scope('Restore-Net',reuse=reuse):
        luma = my_model_pool.U_net2(input,num_down=4,num_block=1,num_conv=1,num_out=12,is_residual=False,fil_s=3,start_chan=32,is_global=False,name='U-Net',reuse=reuse)
        restored = tf.depth_to_space(luma,2)
    return restored
    '''
    '''
    ## Enhance-Net
def EnhanceNet(input,reuse=False):
    restored_gamma = tf.maximum(input,1e-4)**(1.0/2.2)
    with tf.variable_scope('Enhance-Net',reuse=reuse):
        restored_gamma = tf.space_to_depth(restored_gamma,2)
        final = my_model_pool.U_net2(restored_gamma,num_down=4,num_block=1,num_conv=1,num_out=12,is_residual=True,fil_s=3,rate=[1,2,2,4,8],start_chan=32,is_global=True,name='U-Net',reuse=reuse)
        final = tf.depth_to_space(final,2)
    return final

### declare tensors
test_input = tf.placeholder(tf.float32,shape=[None,None,None,4])
test_middle = tf.placeholder(tf.float32,shape=[None,None,None,3])
test_output = tf.placeholder(tf.float32,shape=[None,None,None,3])

### declare network
test_enhan = EnhanceNet(RestoreNet(test_input,reuse=False),reuse=False)
test_enhan_c = tf.clip_by_value(test_enhan,0.0,1.0)
test_output_concat = tf.concat([test_enhan_c,test_output],2)


## metric 
psnr = tf.image.psnr(test_enhan_c,test_output,max_val=1.0)
ssim = tf.image.ssim(test_enhan_c,test_output,max_val=1.0)
sess=tf.Session()
### load network
recons_vars = [v for v in tf.global_variables() if v.name.startswith("Restore-Net")]
enhan_vars = [v for v in tf.global_variables() if v.name.startswith("Enhance-Net")]
saver = tf.train.Saver(recons_vars+enhan_vars,max_to_keep=1000)
saver.restore(sess,ckpt_model_path)
    
if not os.path.isdir("%s/test"%(ckpt)):
    os.makedirs("%s/test"%(ckpt))

psnr_list = np.zeros(len(input_filelist), dtype=np.float32)
ssim_list = np.zeros(len(input_filelist), dtype=np.float32)
for id in range(len(input_filelist)):
    input_id = input_filelist[id]#os.path.splitext(input_files[id])
    input_data = np.fromfile(input_path+'/'+input_id,dtype=np.float32)
    H,W,input_img = np.int32(input_data[0]),np.int32(input_data[1]),np.float32(input_data[2:])
    input_img = np.reshape(input_img,[1,H,W,4])

    output_id = output_filelist[id]
    output_fn = output_path+'/'+output_id
    output_img = np.expand_dims(np.float32(cv2.imread(output_fn,-1))/255.0,axis=0)[:,:,:,::-1]


    feed_dict = {test_input:input_img,test_output:output_img}
    in_run = [psnr,ssim,test_enhan_c]
    out_run = sess.run(in_run,feed_dict=feed_dict)
    psnr_,ssim_,test_enhan_c_ = out_run
  
    ## prepare to output
    psnr_list[id],ssim_list[id] = psnr_,ssim_
    mpimg.imsave("%s/test/%s_out.png"%(ckpt,input_id),test_enhan_c_[0,:,:,:])
    print("Test_image=%d,psnr=%.4f,ssim=%.4f"%(id+1,np.mean(psnr_list[np.where(psnr_list)]),np.mean(ssim_list[np.where(ssim_list)])))

sess.close()
target=open("%s/test/metric.txt"%(ckpt),'w')
target.write("psnr_enh = %.5f\n"%np.mean(psnr_list[np.where(psnr_list)]))
target.write("ssim_enh = %.5f"%np.mean(ssim_list[np.where(ssim_list)]))
target.close()


