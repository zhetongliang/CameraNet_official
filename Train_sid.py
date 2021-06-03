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
os.environ['CUDA_VISIBLE_DEVICES']='0'

    
    
#settings 
base = './Data'
train_input_filelist = base+'/SID/inputlist_train.txt'
train_middle_filelist = base+'/SID/middlelist_train.txt'
train_output_filelist = base+'/SID/outputlist_train.txt'
#train_param_filelist = base+'/Train/paramlist.txt'
test_input_filelist = base+'/SID/inputlist_test.txt'
test_middle_filelist = base+'/SID/middlelist_test.txt'
test_output_filelist = base+'/SID/outputlist_test.txt'
#test_param_filelist = base+'/Test/paramlist.txt'

train_input_path = base+'/SID/input'
train_middle_path = base+'/SID/middle'
train_output_path = base+'/SID/output'
#train_param_path = base+'/Train/parameters'
test_input_path = base+'/SID/input'
test_middle_path = base+'/SID/middle'
test_output_path = base+'/SID/output'
#test_param_path = base+'/Test/parameters'
ckpt = '/Ckpt/CameraNet_SID'

if not os.path.isdir(ckpt):
    os.makedirs(ckpt)

initial_lr = 0.0001
is_training = True
num_epochs = 4800
Train_patchsize = 512
Test_patchsize = 512
num_train_sample = 280
num_test_sample = 93
ckpt_fre = 100
saving_fre = 100
eps = 0.00001

# Variable
lr = tf.placeholder(tf.float32,shape=[])
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



if is_training is True:
    ## Input pipeline

    with tf.variable_scope('train_data'):
        train_image,train_middle,train_label = dp.file_input_pipeline_LSD(train_input_filelist,train_middle_filelist,train_output_filelist,
                                          train_input_path,train_middle_path,train_output_path,
                                          num_epochs=10000,patchsize=512,is_augment=True,is_shuffle=True,num_threads=6)
    with tf.variable_scope('test_data'):
        test_image,test_middle,test_label = dp.file_input_pipeline_LSD(test_input_filelist,test_middle_filelist,test_output_filelist,
                                           test_input_path,test_middle_path,test_output_path,
                                          num_epochs=10000,patchsize=Test_patchsize,is_augment=False,is_shuffle=False,num_threads=1)

    ## Preprocessing ##

    train_middle_lin = tf.pow(tf.clip_by_value(train_middle,0.0,1.0),2.2)
    test_middle_lin = tf.pow(tf.clip_by_value(test_middle,0.0,1.0),2.2) 

    ##### Building Network ##### 
    train_recons = RestoreNet(train_image,reuse=False)
    train_enhan_sep = EnhanceNet(train_middle_lin,reuse=False)
    train_enhan = EnhanceNet(train_recons,reuse=True)
    
    test_recons = RestoreNet(test_image,reuse=True)
    test_enhan = EnhanceNet(test_recons,reuse=True)

    #### Postprocessing ##
    test_recons_c = tf.clip_by_value(test_recons,0.0,1.0)
    test_enhan_c = tf.clip_by_value(test_enhan,0.0,1.0)
    test_check_img_recons = tf.pow(tf.concat([test_recons_c,test_middle_lin],2),1.0/2.2)
    test_check_img_enhan = tf.concat([test_enhan_c,test_label],2)

    ##### Loss and metrics ####### 
    train_restore_loss_lin = tf.reduce_mean(tf.abs(train_recons-train_middle_lin))
    train_restore_loss_log = tf.reduce_mean(tf.abs(tf.log(tf.maximum(train_recons,0.0)+1e-4) 
                                - tf.log(tf.maximum(train_middle_lin,0.0)+1e-4)))

    train_restore_loss = train_restore_loss_lin + train_restore_loss_log
    train_enhance_sep_loss = tf.reduce_mean(tf.abs(train_enhan_sep-train_label))
    train_enhance_loss = tf.reduce_mean(tf.abs(train_enhan-train_label))
    train_joint_loss = train_enhance_loss*0.1+train_restore_loss*0.9

    ##### Metric ######
    psnr_recons = tf.image.psnr(test_recons_c,test_middle_lin,max_val=1.0)
    psnr_enhan = tf.image.psnr(test_enhan_c,test_label,max_val=1.0)
    ssim_recons = tf.image.ssim(test_recons_c,test_middle_lin,max_val=1.0)
    ssim_enhan = tf.image.ssim(test_enhan_c,test_label,max_val=1.0)

    ##  Optimizer
    recons_vars = [v for v in tf.global_variables() if v.name.startswith("Restore-Net")]
    enhan_vars = [v for v in tf.global_variables() if v.name.startswith("Enhance-Net")]
    opt_recons=tf.train.AdamOptimizer(learning_rate=lr,epsilon=1e-4).minimize(train_restore_loss,var_list=recons_vars)
    opt_enhan=tf.train.AdamOptimizer(learning_rate=lr,epsilon=1e-4).minimize(train_enhance_sep_loss,var_list=enhan_vars)
    opt_all=tf.train.AdamOptimizer(learning_rate=lr,epsilon=1e-4).minimize(train_joint_loss,var_list=recons_vars+enhan_vars)

    ######## Start training #####
    sess=tf.Session()

    # Saver to save network model
    saver_recons = tf.train.Saver(recons_vars,max_to_keep=1000)
    saver_enh = tf.train.Saver(enhan_vars,max_to_keep=1000)
    saver_all=tf.train.Saver(recons_vars+enhan_vars,max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    restart = 0
    #saver_recons.restore(sess,ckpt + '/' + 'restore' + '_%04d/model.ckpt'%(4000))


    sess.run(tf.local_variables_initializer()) 


    ################ Training ####################
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    Trainlist0 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist1 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist2 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist3 = np.zeros(num_train_sample, dtype=np.float32)
    Trainlist4 = np.zeros(num_train_sample, dtype=np.float32)
    for epoch in range(restart+1,num_epochs+1):
        cnt=0
        
        if epoch<=4000:      
            learning_rate = initial_lr*(0.1**(np.float32(epoch-1)/3000.0))
            opt = opt_recons
            psnr,ssim = psnr_recons,ssim_recons
            test_check_img = test_check_img_recons
            saver = saver_recons
            stage = 'restore'
            for id in range(num_train_sample):         
                st=time.time()
                train_ops = [train_restore_loss_lin,train_restore_loss_log,opt]
                out_ops =sess.run(train_ops,feed_dict={lr:learning_rate})          
                Trainlist0[id] = out_ops[0]
                Trainlist1[id] = out_ops[1]
                print("cameranet_sid: epoch=%d, count=%d, rest_lin=%.4f,rest_log=%.4f,time=%.3f"%(epoch,cnt,np.mean(Trainlist0[np.where(Trainlist0)]),
                                                                                                                    np.mean(Trainlist1[np.where(Trainlist1)]),time.time()-st))
                cnt+=1
        elif epoch>4000 and epoch<=4500:
            learning_rate = initial_lr*(0.1**(np.float32(epoch-4000)/300.0))
            opt = opt_enhan
            psnr,ssim = psnr_enhan,ssim_enhan
            test_check_img = test_check_img_enhan
            saver = saver_enh
            stage = 'enhance'
            for id in range(num_train_sample):         
                st=time.time()
                train_ops = [train_enhance_sep_loss,opt]
                out_ops =sess.run(train_ops,feed_dict={lr:learning_rate})          
                Trainlist2[id] = out_ops[0]
                print("cameranet_sid: epoch=%d, count=%d, enh=%.4f,time=%.3f"%(epoch,cnt,np.mean(Trainlist2[np.where(Trainlist2)]),time.time()-st))
                cnt+=1
        else:
            learning_rate = 0.5*initial_lr*(0.1**(np.float32(epoch-4500)/200.0))
            opt = opt_all
            psnr,ssim = psnr_enhan,ssim_enhan
            test_check_img = test_check_img_enhan
            saver = saver_all
            stage = 'all'
            for id in range(num_train_sample):         
                st=time.time()
                train_ops = [train_enhance_loss,opt]
                out_ops =sess.run(train_ops,feed_dict={lr:learning_rate})          
                Trainlist3[id] = out_ops[0]
                print("cameranet_sid: epoch=%d, count=%d, enh=%.4f,time=%.3f"%(epoch,cnt,np.mean(Trainlist3[np.where(Trainlist3)]),time.time()-st))
                cnt+=1
        
        
    ## model checkpoint
        if stage == 'restore':           
            psnr,ssim = psnr_recons,ssim_recons
            test_check_img = test_check_img_recons
            saver = saver_recons
        elif stage == 'enhance':
            psnr,ssim = psnr_enhan,ssim_enhan
            test_check_img = test_check_img_enhan
            saver = saver_enh
        else:
            psnr,ssim = psnr_enhan,ssim_enhan
            test_check_img = test_check_img_enhan
            saver = saver_all
    
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

