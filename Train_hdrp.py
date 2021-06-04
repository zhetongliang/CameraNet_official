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



#settings 
base = 'Data/HDRp'

train_input_filelist = base+'/Train/inputlist.txt'
train_middle_filelist = base+'/Train/middlelist.txt'
train_output_filelist = base+'/Train/outputlist.txt'
train_input_path = base+'/Train/input'
train_middle_path = base+'/Train/middle'
train_output_path = base+'/Train/output'

test_input_filelist = base+'/Test/inputlist.txt'
test_middle_filelist = base+'/Test/middlelist.txt'
test_output_filelist = base+'/Test/outputlist.txt'
test_input_path = base+'/Test/input'
test_middle_path = base+'/Test/middle'
test_output_path = base+'/Test/output'


ckpt = 'Ckpt/CameraNet_HDRp'

if not os.path.isdir(ckpt):
    os.makedirs(ckpt)

initial_lr = 0.0001
is_training=True
num_epochs = 1800
num_train_sample = 675
num_test_sample = 247
ckpt_fre = 200
eps = 0.00001


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

def clip(x,low=0.0,high=1.0):
    return tf.clip_by_value(x,low,high)



def restore_loss_define(train_recons,train_middle_lin):
    train_restore_loss_lin = tf.reduce_mean(tf.abs(train_recons-train_middle_lin))
    train_restore_loss_log = tf.reduce_mean(tf.abs(tf.log(tf.maximum(train_recons,0.0)+0.001) - tf.log(train_middle_lin+0.001)))

    train_restore_loss = train_restore_loss_lin + train_restore_loss_log
    return train_restore_loss,train_restore_loss_lin



if is_training is True:
    ## Input pipeli
    with tf.variable_scope('train_data'):
        train_image,train_middle,train_output = dp.file_input_pipeline(train_input_filelist,train_middle_filelist,train_output_filelist,
                                          train_input_path,train_middle_path,train_output_path,
                                          num_epochs=10000,patchsize=512,batchsize=1,is_augment=True,is_shuffle=True,num_threads=5)
    with tf.variable_scope('test_data'):
        test_image,test_middle,test_output  = dp.file_input_pipeline(test_input_filelist,test_middle_filelist,test_output_filelist,
                                          test_input_path,test_middle_path,test_output_path,
                                          num_epochs=10000,patchsize=512,batchsize=1,is_augment=True,is_shuffle=True,num_threads=1)


    ##### Building Network ##### 
    train_recons = RestoreNet(train_image,reuse=False)
    train_enhan_enh = EnhanceNet(train_middle,reuse=False)
    train_enhan_all = EnhanceNet(train_recons,reuse=True)
    
    test_recons = RestoreNet(test_image,reuse=True)
    test_enhan = EnhanceNet(test_recons,reuse=True)

    ##### Loss and metrics ####### 
    train_restore_loss,train_l1_loss = restore_loss_define(train_recons,train_middle)
    train_enhance_loss_s1 = tf.reduce_mean(tf.abs(train_enhan_enh-train_output)) 
    train_enhance_loss_s2 = tf.reduce_mean(tf.abs(train_enhan_all-train_output))
    train_all_loss_s2 = train_enhance_loss_s1 + train_enhance_loss_s2
    
    ##### Post processing ####
    test_recons_c = clip(test_recons)
    test_enhan_c = clip(test_enhan)
    test_check_img_recons = tf.pow(tf.concat([test_recons_c,test_middle],2),1.0/2.2)
    test_check_img_enhan = tf.concat([test_enhan_c,test_output],2)

    ##### Metric ######
    psnr_recons = tf.image.psnr(test_recons,test_middle,max_val=1.0)
    psnr_enhan = tf.image.psnr(test_enhan,test_output,max_val=1.0)
    ssim_recons = tf.image.ssim(test_recons,test_middle,max_val=1.0)
    ssim_enhan = tf.image.ssim(test_enhan,test_output,max_val=1.0)

    ##  Optimizer
    recons_vars = [v for v in tf.global_variables() if v.name.startswith("Restore-Net")]
    enhan_vars = [v for v in tf.global_variables() if v.name.startswith("Enhance-Net")]
    opt_restore = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5,epsilon=1e-04).minimize(train_restore_loss,var_list=recons_vars)
    opt_enh = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5,epsilon=1e-04).minimize(train_enhance_loss_s1,var_list=enhan_vars)   
    opt_all = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5,epsilon=1e-04).minimize(train_all_loss_s2,var_list=recons_vars+enhan_vars)

    ######## Start training #####
    sess=tf.Session()

    # Saver to save network model
    saver_all = tf.train.Saver(recons_vars+enhan_vars,max_to_keep=1000)
    saver_restore = tf.train.Saver(recons_vars,max_to_keep=1000)
    saver_enhance = tf.train.Saver(enhan_vars,max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    restart = 0
    #saver_enhance.restore(sess,'/home/justin/Desktop/8T1/old4t/Experiments/single/CameraNet/new_model/CameraNet_HDR_progressive_split_sim/enh/ckpt/0300_ckpt/model.ckpt')
    #saver_restore.restore(sess,'/home/justin/Desktop/8T1/old4t/Experiments/single/CameraNet/new_model/CameraNet_HDR_progressive_split_sim/enh_vgg_imgnet/ckpt/0120_ckpt/model.ckpt')
    #saver.restore(sess,ckpt+'/0410_ckpt/model.ckpt')
 
    sess.run(tf.local_variables_initializer()) 


    ################ Training ####################
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    Trainlist0 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist1 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist2 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist3 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist4 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist5 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist6 = np.zeros(num_train_sample, dtype=np.float32)
    for epoch in range(restart+1,num_epochs+1):
        cnt = 0

        if epoch<=1000:      
            learning_rate = initial_lr*(0.1**(np.float32(epoch-1)/800))     
            psnr,ssim = psnr_recons,ssim_recons
            test_check_img = test_check_img_recons
            saver = saver_restore
            stage = 'restore'
            for id in range(num_train_sample):              
                st=time.time()
                opts = [train_l1_loss,opt_restore] 
                out_ops =sess.run(opts,feed_dict={lr:learning_rate})    
                Trainlist0[id] = out_ops[0]
                show0 = np.mean(Trainlist0[np.where(Trainlist0)])
                print("cameranet_hdrp restore: epoch=%d, count=%d, restore l1=%.4f, time=%.3f"%(epoch,cnt,show0,time.time()-st))
                cnt+=1
            
        elif epoch>1000 and epoch<=1500:   
            learning_rate = initial_lr*(0.1**(np.float32(epoch-1001)/300.0))   
            psnr,ssim = psnr_enhan,ssim_enhan
            test_check_img = test_check_img_enhan
            saver = saver_enhance
            stage = 'enhance'
            for id in range(num_train_sample):         
                st=time.time()
                opts = [train_enhance_loss_s1,opt_enh]
                out_ops =sess.run(opts,feed_dict={lr:learning_rate})          
                Trainlist1[id] = out_ops[0]
                show1 = np.mean(Trainlist1[np.where(Trainlist1)])
                print("cameranet_hdrp enh: epoch=%d, count=%d, enh_l1=%.4f,time=%.3f"%(epoch,cnt,show1,time.time()-st))
                cnt+=1
                
        elif epoch>1500 and epoch<=1800:   
            learning_rate = 0.5*initial_lr*(0.1**(np.float32(epoch-1500)/200.0))  
            psnr,ssim = psnr_enhan,ssim_enhan
            test_check_img = test_check_img_enhan
            saver = saver_all
            stage = 'all'
            for id in range(num_train_sample):         
                st=time.time()
                opts = [train_enhance_loss_s2,opt_all]
                out_ops =sess.run(opts,feed_dict={lr:learning_rate})          
                Trainlist2[id] = out_ops[0]
                show2 = np.mean(Trainlist2[np.where(Trainlist2)])
                print("cameranet_hdrp all: epoch=%d, count=%d, enh_l1=%.4f,time=%.3f"%(epoch,cnt,show2,time.time()-st))
                cnt+=1

    
        save_path = ckpt + '/' + stage + '_%04d'%(epoch)
        save_img_path = save_path + '/results'
        if epoch%ckpt_fre==0:
            
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            saver.save(sess,save_path+"/model.ckpt")
            
            if not os.path.isdir(save_img_path):
                os.makedirs(save_img_path)
            SSIMlist = np.zeros(num_test_sample, dtype=np.float32)
            PSNRlist = np.zeros(num_test_sample, dtype=np.float32)
            for ind in range(num_test_sample):
                st = time.time()
                test_ops = [ssim,psnr,test_check_img]
                out_ops = sess.run(test_ops)   
                print("Test_image=%d, time=%.3f"%(ind+1,time.time()-st))
                SSIMlist[ind] = out_ops[0]
                PSNRlist[ind] = out_ops[1]
                test_check_img_ = out_ops[2]
                test_check_img_ = np.clip(test_check_img_,0.0,1.0)
                mpimg.imsave(save_img_path+"/%04d.png"%(ind),test_check_img_[0,:,:,:])

            target=open(save_path+"/metric.txt",'w')
            target.write("SSIM = %.5f\n"%np.mean(SSIMlist[np.where(SSIMlist)]))
            target.write("PNSR = %.5f\n"%np.mean(PSNRlist[np.where(PSNRlist)]))
            target.close() 
            
    coord.request_stop()
    coord.join(threads)
    sess.close()

