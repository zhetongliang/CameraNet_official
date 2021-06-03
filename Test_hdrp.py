#  Archi: global, noise map. training: two stage
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
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import csv,cv2,glob

# Variable
lr = tf.placeholder(tf.float32,shape=[])
lr_D = tf.placeholder(tf.float32,shape=[])
factor_re = tf.placeholder(tf.float32,shape=[])
factor_ce = tf.placeholder(tf.float32,shape=[])



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



def nor(x):
    return (x-tf.reduce_min(x,[1,2,3],keepdims=True))/(tf.reduce_max(x,[1,2,3],keepdims=True)-tf.reduce_min(x,[1,2,3],keepdims=True))
ckpt = 'Ckpt/CameraNet_HDRp'



source_path = 'Data/HDRp/Test/'
input_path = source_path + 'input'
output_path = source_path + 'output'
input_files = os.listdir(input_path)
output_files = os.listdir(output_path)

### declare tensors
test_input = tf.placeholder(tf.float32,shape=[1,None,None,4])
test_output = tf.placeholder(tf.float32,shape=[None,None,None,3])

### declare network
test_recons = RestoreNet(test_input,reuse=False)
test_enhan = EnhanceNet(test_recons,reuse=False)
recons_vars = [v for v in tf.global_variables() if v.name.startswith("Restore-Net")]
enhan_vars = [v for v in tf.global_variables() if v.name.startswith("Enhance-Net")]

### Post-process
test_enhan_c = tf.clip_by_value(test_enhan,0.0,1.0)

## metric 
psnr_enh = tf.image.psnr(test_enhan_c,test_output,max_val=1.0)
ssim_enh = tf.image.ssim(test_enhan_c,test_output,max_val=1.0)

sess=tf.Session()
### load network
saver=tf.train.Saver(max_to_keep=1000)
#saver_enhance=tf.train.Saver(enhan_vars,max_to_keep=1000)
saver.restore(sess,ckpt+'/all_1800/model.ckpt')
#saver_restore.restore(sess,'/home/justin/Desktop/8T1/old4t/Experiments/single/CameraNet/new_model/CameraNet_HDR_progressive_split_sim/no_dilated/ckpt/1000_ckpt/model.ckpt')
    
if not os.path.isdir("%s/test"%(ckpt)):
    os.makedirs("%s/test"%(ckpt))

psnr_list = np.zeros(len(input_files), dtype=np.float32)
ssim_list = np.zeros(len(input_files), dtype=np.float32)

for id in range(len(input_files)):

    input_fn = input_files[id]
    test_id = input_fn[0:4]
    input_data = np.fromfile(input_path+'/'+input_fn,dtype=np.float32)
    H,W,input_img = np.int32(input_data[0])//2,np.int32(input_data[1])//2,np.float32(input_data[2:])
    input_img = np.reshape(input_img,[1,H,W,4])


    #middle_img = middle_img[:,y*2:y*2+ps*2,x*2:x*2+ps*2,:]

    output_fn = glob.glob(output_path+'/'+test_id+'*')
    output_fn = output_fn[0]
    output_img = np.expand_dims(np.float32(cv2.imread(output_fn,-1))/255.0,axis=0)[:,:,:,::-1]
#output_img = output_img[:,y*2:y*2+ps*2,x*2:x*2+ps*2,:]


    feed_dict = {test_input:input_img,test_output:output_img}
    in_run = [psnr_enh,ssim_enh,test_enhan_c]
    out_run = sess.run(in_run,feed_dict=feed_dict)	
    psnr_enh_,ssim_enh_,test_enhan_c_ = out_run
  
    psnr_list[id],ssim_list[id] = psnr_enh_,ssim_enh_

    #mpimg.imsave("%s/test/%s_restore.png"%(ckpt,test_id),test_recons_c_[0,:,:,:])
    #mpimg.imsave("%s/test/%s_restore_gt.png"%(ckpt,test_id),test_middle_c_[0,:,:,:])
    mpimg.imsave("%s/test/%s_enhance.png"%(ckpt,test_id),test_enhan_c_[0,:,:,:])
    #mpimg.imsave("%s/test/%s_enhance_gt.png"%(ckpt,test_id),test_output_[0,:,:,:])	

    print("Test_image=%d"%(id+1))

sess.close()
target=open("%s/test/metric.txt"%(ckpt),'w')
target.write("psnr = %.5f\n"%np.mean(psnr_list[np.where(psnr_list)]))
target.write("ssim = %.5f\n"%np.mean(ssim_list[np.where(ssim_list)]))
target.close()


