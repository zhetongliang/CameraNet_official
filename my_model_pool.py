import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
sys.path.append('../')
import layers

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat_c(x1, x2, output_channels, in_channels, scope,reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        pool_size = 2
        deconv_filter = tf.get_variable(shape= [pool_size, pool_size, output_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.001),name='dcf')
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

        deconv_output =  tf.concat([deconv, x2],3)
   #     deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def est_structure(x,size,sigma):
    ## x is a single channel tensor
    def _tf_fspecial_gauss(size, sigma):
        x_data, y_data = np.mgrid[-size//2 - 1:size//2 + 1, -size//2 - 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2.0 + y**2.0)/(2.0*sigma**2.0)))
        return g / tf.reduce_sum(g)

    window = _tf_fspecial_gauss(size, sigma)
    final = tf.nn.conv2d(x, window, strides=[1,1,1,1], padding='SAME')
    return final

def U_net(input,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,is_residual=False,start_chan=32,is_global=False,name=None,reuse=False):
    ## parameters
    act = lrelu
    conv_ = []
    chan_ = []
    for i in range(num_down+1):
        chan_.append(start_chan*(2**(i)))
    rate_ = []


    with tf.variable_scope(name,reuse=reuse):
        current = input
        with tf.variable_scope('contracting_ops',reuse=reuse):
            for i in range(num_down):
                current = slim.conv2d(current,chan_[i],[fil_s,fil_s], activation_fn=act,scope='g_conv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                         current=slim.conv2d(current,chan_[i],[fil_s,fil_s], rate=rate[i], activation_fn=act,scope='g_conv%d_block%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(current,chan_[i],[fil_s,fil_s],  activation_fn=None,scope='g_conv%d_block%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)            
                pool=slim.max_pool2d(current, [fil_s, fil_s], padding='SAME',scope='pool%d'%(i))
                conv_.append(current)
                current = pool

            current=slim.conv2d(current,chan_[num_down],[fil_s,fil_s],  activation_fn=act,scope='g_conv%d'%(num_down),reuse=reuse)
            contract_temp = current
        ##
        with tf.variable_scope('local_ops',reuse=reuse):
            current = contract_temp
            for ii in range(num_block):
                adding = current
                for j in range(num_conv):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s], rate=rate[num_down],activation_fn=act,scope='g_conv_block%d_%d'%(ii,j),reuse=reuse)
                current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s], activation_fn=None,scope='g_conv_block%d'%(ii),reuse=reuse)
                if is_residual is True:                
                    current = act(current + adding)
            restore_temp = current

        if is_global is True:
            with tf.variable_scope('global_ops',reuse=reuse):
                current = contract_temp
                '''
                for i in range(3):
                       current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],activation_fn=act,scope='global%d'%(i),reuse=reuse)
            current = slim.max_pool2d(current, [2, 2], padding='SAME',scope='global_pool%d'%(i))
        '''
                global_feature = tf.reduce_mean(current,[1,2],keepdims=False)
                current = slim.fully_connected(global_feature,chan_[num_down]*2,activation_fn=act,scope='fully_enhan00',reuse=reuse)
                current = slim.fully_connected(current,chan_[num_down]*2,activation_fn=act,scope='fully_enhan01',reuse=reuse)
                global_feature = tf.reshape(current,[-1,1,1,chan_[num_down]*2])
            restore_temp = act(restore_temp*global_feature[:,:,:,0:chan_[num_down]] + global_feature[:,:,:,chan_[num_down]:])

        with tf.variable_scope('expanding_ops',reuse=reuse):
            current = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(current,chan_[index_current],[fil_s,fil_s],rate=rate[index_current], activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                        current=slim.conv2d(current,  chan_[index_current],[fil_s,fil_s], rate=rate[index_current], activation_fn=act,scope='g_dconv_block%d%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(current,  chan_[index_current],[fil_s,fil_s],  activation_fn=None,scope='g_dconv_block%d%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)
            final = slim.conv2d(current,  num_out,[1,1],  activation_fn=None,scope='final',reuse=reuse)

    return final

def pad(x,p=1):
    return tf.pad(x,[[0,0],[p,p],[p,p],[0,0]],'REFLECT')
def U_net2(input,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,is_residual=False,start_chan=32,act=lrelu,is_global=False,name=None,reuse=False):
    ## parameters
    conv_ = []
    chan_ = []
    for i in range(num_down+1):
        chan_.append(start_chan*(2**(i)))
    
    with tf.variable_scope(name,reuse=reuse):
        current = input
        with tf.variable_scope('contracting_ops',reuse=reuse):
            for i in range(num_down):
                current = slim.conv2d(pad(current),chan_[i],[fil_s,fil_s], activation_fn=act,scope='g_conv%d'%(i),padding='VALID',reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                         current=slim.conv2d(pad(current,rate[i]),chan_[i],[fil_s,fil_s], rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv%d_block%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(pad(current,rate[i]),chan_[i],[fil_s,fil_s],  activation_fn=None, rate=rate[i],padding='VALID',scope='g_conv%d_block%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)     
                    else:
                        current = act(current)    
                pool=slim.max_pool2d(current, [fil_s, fil_s], padding='SAME',scope='pool%d'%(i))
                conv_.append(current)
                current = pool

            current=slim.conv2d(pad(current),chan_[num_down],[fil_s,fil_s],  activation_fn=act,padding='VALID',scope='g_conv%d'%(num_down),reuse=reuse)
            contract_temp = current
        ##
        with tf.variable_scope('local_ops',reuse=reuse):
            current = contract_temp
            for ii in range(num_block):
                adding = current
                for j in range(num_conv):
                    current = slim.conv2d(pad(current,rate[num_down]),chan_[num_down],[fil_s,fil_s], rate=rate[num_down],activation_fn=act,padding='VALID',scope='g_conv_block%d_%d'%(ii,j),reuse=reuse)
                current = slim.conv2d(pad(current),chan_[num_down],[fil_s,fil_s], activation_fn=None,padding='VALID',scope='g_conv_block%d'%(ii),reuse=reuse)
                if is_residual is True:                
                    current = act(current + adding)
                else:
                    current = act(current) 
            restore_temp = current

        if is_global is True:
            with tf.variable_scope('global_ops',reuse=reuse):
                current = contract_temp
                '''
                for i in range(3):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],activation_fn=act,scope='global%d'%(i),reuse=reuse)
                    current = slim.max_pool2d(current, [2, 2], padding='SAME',scope='global_pool%d'%(i))
                '''
                global_feature = tf.reduce_mean(current,[1,2],keepdims=False)
                current = slim.fully_connected(global_feature,chan_[num_down]*2,activation_fn=lrelu,scope='fully_enhan00',reuse=reuse)
                current = slim.fully_connected(current,chan_[num_down]*2,activation_fn=None,scope='fully_enhan01',reuse=reuse)
                global_feature = tf.reshape(current,[-1,1,1,chan_[num_down]*2])
            restore_temp = act(restore_temp*global_feature[:,:,:,0:chan_[num_down]] + global_feature[:,:,:,chan_[num_down]:])

        with tf.variable_scope('expanding_ops',reuse=reuse):
            current = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(pad(current),chan_[index_current],[fil_s,fil_s], padding='VALID',activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                        current=slim.conv2d(pad(current,rate[index_current]),  chan_[index_current],[fil_s,fil_s], rate=rate[index_current], padding='VALID',activation_fn=act,scope='g_dconv_block%d%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(pad(current),  chan_[index_current],[fil_s,fil_s],  padding='VALID',activation_fn=None,scope='g_dconv_block%d%d'%(i,ii),reuse=reuse)
                    if i == num_down-1 and ii == num_block-1:
                        if is_residual is True:
                            current = current + adding
                        else:
                            current = current
                    else:
                        if is_residual is True:
                            current = act(current + adding)
                        else:
                            current = act(current) 
            final = slim.conv2d(current,  num_out,[1,1],  activation_fn=None,scope='final',reuse=reuse)

    return final

def IN(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.get_variable(shape=[1,1,1,channels],initializer=tf.constant_initializer(0.0),name='shift')
    scale = tf.get_variable(shape=[1,1,1,channels],initializer=tf.constant_initializer(1.0),name='scale')

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def U_net3(input,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,is_IN=False,is_residual=False,start_chan=32,act=lrelu,is_global=False,name=None,reuse=False):
    ## parameters
    conv_ = []
    chan_ = []
    for i in range(num_down+1):
        chan_.append(start_chan*(2**(i)))
    if is_IN == True:
        nf = IN
    else:
        nf = None
    
    with tf.variable_scope(name,reuse=reuse):
        current = input
        with tf.variable_scope('contracting_ops',reuse=reuse):
            for i in range(num_down):
                current = slim.conv2d(pad(current),chan_[i],[fil_s,fil_s], activation_fn=act,scope='g_conv%d'%(i),padding='VALID',reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                         current=slim.conv2d(pad(current,rate[i]),chan_[i],[fil_s,fil_s], normalizer_fn=nf,rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv%d_block%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(pad(current,rate[i]),chan_[i],[fil_s,fil_s],  normalizer_fn=nf,activation_fn=None, rate=rate[i],padding='VALID',scope='g_conv%d_block%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)     
                    else:
                        current = act(current)    
                pool=slim.max_pool2d(current, [fil_s, fil_s], padding='SAME',scope='pool%d'%(i))
                conv_.append(current)
                current = pool

            current=slim.conv2d(pad(current),chan_[num_down],[fil_s,fil_s], normalizer_fn=nf, activation_fn=act,padding='VALID',scope='g_conv%d'%(num_down),reuse=reuse)
            contract_temp = current
        ##
        with tf.variable_scope('local_ops',reuse=reuse):
            current = contract_temp
            for ii in range(num_block):
                adding = current
                for j in range(num_conv):
                    current = slim.conv2d(pad(current,rate[num_down]),chan_[num_down],[fil_s,fil_s], normalizer_fn=nf,rate=rate[num_down],activation_fn=act,padding='VALID',scope='g_conv_block%d_%d'%(ii,j),reuse=reuse)
                current = slim.conv2d(pad(current),chan_[num_down],[fil_s,fil_s], normalizer_fn=nf,activation_fn=None,padding='VALID',scope='g_conv_block%d'%(ii),reuse=reuse)
                if is_residual is True:                
                    current = act(current + adding)
                else:
                    current = act(current) 
            restore_temp = current
            
        if is_global is True:
            with tf.variable_scope('global_ops',reuse=reuse):
                current = contract_temp
                '''
                for i in range(3):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],activation_fn=act,scope='global%d'%(i),reuse=reuse)
                    current = slim.max_pool2d(current, [2, 2], padding='SAME',scope='global_pool%d'%(i))
                '''
                global_feature = tf.reduce_mean(current,[1,2],keepdims=False)
                current = slim.fully_connected(global_feature,chan_[num_down]*2,activation_fn=lrelu,scope='fully_enhan00',reuse=reuse)
                current = slim.fully_connected(current,chan_[num_down]*2,activation_fn=None,scope='fully_enhan01',reuse=reuse)
                global_feature = tf.reshape(current,[-1,1,1,chan_[num_down]*2])
            restore_temp = act(restore_temp*global_feature[:,:,:,0:chan_[num_down]] + global_feature[:,:,:,chan_[num_down]:])

        with tf.variable_scope('expanding_ops',reuse=reuse):
            current = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(pad(current),chan_[index_current],[fil_s,fil_s], padding='VALID',normalizer_fn=nf,activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                        current=slim.conv2d(pad(current,rate[index_current]),  chan_[index_current],[fil_s,fil_s], rate=rate[index_current], padding='VALID',activation_fn=act,scope='g_dconv_block%d%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(pad(current),  chan_[index_current],[fil_s,fil_s], normalizer_fn=nf, padding='VALID',activation_fn=None,scope='g_dconv_block%d%d'%(i,ii),reuse=reuse)
                    if i == num_down-1 and ii == num_block-1:
                        if is_residual is True:
                            current = current + adding
                        else:
                            current = current
                    else:
                        if is_residual is True:
                            current = act(current + adding)
                        else:
                            current = act(current) 
            final = slim.conv2d(current,  num_out,[1,1],  activation_fn=None,scope='final',reuse=reuse)

    return final

def U_net1(input,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,is_residual=False,start_chan=32,act=lrelu,is_global=False,name=None,reuse=False):
    ## parameters
    conv_ = []
    chan_ = []
    for i in range(num_down+1):
        chan_.append(start_chan*(2**(i)))

    with tf.variable_scope(name,reuse=reuse):
        current = input
        with tf.variable_scope('contracting_ops',reuse=reuse):
            for i in range(num_down):
                current = slim.conv2d(current,chan_[i],[fil_s,fil_s],activation_fn=act,scope='g_conv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                         current=slim.conv2d(current,chan_[i],[fil_s,fil_s], rate=rate[i], activation_fn=act,scope='g_conv%d_block%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(current,chan_[i],[fil_s,fil_s], rate=rate[i], activation_fn=None,scope='g_conv%d_block%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)     
                    else:
                        current = act(current)    
                pool=slim.max_pool2d(current, [fil_s, fil_s], padding='SAME',scope='pool%d'%(i))
                conv_.append(current)
                current = pool

            current=slim.conv2d(current,chan_[num_down],[fil_s,fil_s],  activation_fn=act,scope='g_conv%d'%(num_down),reuse=reuse)
            contract_temp = current
        ##
        with tf.variable_scope('local_ops',reuse=reuse):
            current = contract_temp
            for ii in range(num_block):
                adding = current
                for j in range(num_conv):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s], rate=rate[num_down],activation_fn=act,scope='g_conv_block%d_%d'%(ii,j),reuse=reuse)
                current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s], activation_fn=None,scope='g_conv_block%d'%(ii),reuse=reuse)
                if is_residual is True:                
                    current = act(current + adding)
                else:
                    current = act(current) 
            restore_temp = current

        if is_global is True:
            with tf.variable_scope('global_ops',reuse=reuse):
                current = contract_temp
                '''
                for i in range(3):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],activation_fn=act,scope='global%d'%(i),reuse=reuse)
                    current = slim.max_pool2d(current, [2, 2], padding='SAME',scope='global_pool%d'%(i))
                '''
                global_feature = tf.reduce_mean(current,[1,2],keepdims=False)
                current = slim.fully_connected(global_feature,chan_[num_down]*2,activation_fn=lrelu,scope='fully_enhan00',reuse=reuse)
                current = slim.fully_connected(current,chan_[num_down]*2,activation_fn=None,scope='fully_enhan01',reuse=reuse)
                global_feature = tf.reshape(current,[-1,1,1,chan_[num_down]*2])
            restore_temp = act(restore_temp*global_feature[:,:,:,0:chan_[num_down]] + global_feature[:,:,:,chan_[num_down]:])

        with tf.variable_scope('expanding_ops',reuse=reuse):
            current = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(current,chan_[index_current],[fil_s,fil_s], activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                        current=slim.conv2d(current,  chan_[index_current],[fil_s,fil_s], rate=rate[index_current], activation_fn=act,scope='g_dconv_block%d%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(current,  chan_[index_current],[fil_s,fil_s],  activation_fn=None,scope='g_dconv_block%d%d'%(i,ii),reuse=reuse)
                    if i == num_down-1 and ii == num_block-1:
                        if is_residual is True:
                            current = current + adding
                        else:
                            current = current
                    else:
                        if is_residual is True:
                            current = act(current + adding)
                        else:
                            current = act(current) 
            final = slim.conv2d(current,  num_out,[1,1],  activation_fn=None,scope='final',reuse=reuse)

    return final


def CameraNet(input,reuse=False):  

    luma = tf.reduce_mean(input,3,keepdims=True)/3.0
    chroma = input - luma
    input_aug = tf.concat([input,luma,chroma],3)

    with tf.variable_scope('CameraNet_restore',reuse=reuse):
        ## define u-net for luma details
        detail_recovered = U_net(input_aug,num_down=4,num_block=1,num_conv=1,num_out=4,start_chan=32,is_global=False,name='detail_est',reuse=reuse)
        detail_recovered = tf.depth_to_space(detail_recovered,2)

        ## define u-net for luma structures
        structure_recovered = U_net(input_aug,num_down=2,num_block=1,num_conv=1,num_out=4,start_chan=24,is_global=False,name='structure_est',reuse=reuse)
        structure_recovered = tf.depth_to_space(structure_recovered,2)

        ## define u-net for chroma
        rate = [1,4,8]
        chroma_recovered = U_net(input_aug,num_down=2,num_block=1,num_conv=3,num_out=12,rate=rate,start_chan=24,is_global=True,name='chroma_est',reuse=reuse)
        chroma_recovered = tf.depth_to_space(chroma_recovered,2)

        ## combine
        final = structure_recovered + detail_recovered + chroma_recovered

    return final,detail_recovered,structure_recovered,chroma_recovered



def BilateralNet(input,spatial_bin=128,intensity_bin=8,is_glob_pool=True,net_input_size=512,coef=12,last_chan=96,reuse=False):  
    ## Preprocessing 
    act = lrelu
    with tf.variable_scope('Enhancement',reuse=reuse):
        shape = tf.shape(input)
        if is_glob_pool==True:
            H,W = tf.cast(tf.round(shape[1]/6),tf.int32),tf.cast(tf.round(shape[2]/6),tf.int32)
        else:
            H,W = 512,512
        start = tf.image.resize_images(input,[H,W])

        with tf.variable_scope('splat',reuse=reuse):
            n_ds_layers = int(np.log2(net_input_size/spatial_bin))
            current = start
            for i in range(n_ds_layers):
                chan = 32*(2**(i))
                current = slim.conv2d(current,chan,[3,3], stride=1, activation_fn=act,scope='conv_%d'%(i),reuse=reuse)
                current = slim.conv2d(current,chan,[3,3], stride=2, activation_fn=act,scope='conv%d'%(i),reuse=reuse)
            splat_out = current

        with tf.variable_scope('global',reuse=reuse):
            current = splat_out
            for i in range(2):
                current = slim.conv2d(current,64,[3,3], stride=2, activation_fn=act,scope='conv%d'%(i),reuse=reuse)
            _, lh, lw, lc = current.get_shape().as_list()
            if is_glob_pool == False:
                current = tf.reshape(current, [-1, lh*lw*lc])  # flattening
            else:
                current = tf.reduce_mean(current,[1,2],keepdims=False)

            current = slim.fully_connected(current,256,normalizer_fn=None,activation_fn=act,scope='fully_rest00',reuse=reuse)
            current = slim.fully_connected(current,128,normalizer_fn=None,activation_fn=act,scope='fully_rest01',reuse=reuse)
            current = slim.fully_connected(current,last_chan,normalizer_fn=None,activation_fn=act,scope='fully_rest02',reuse=reuse)
            current = tf.reshape(current,[-1,1,1,last_chan])
            global_out = current

        with tf.variable_scope('local',reuse=reuse):
            for i in range(2):
                current = slim.conv2d(current,last_chan,[3,3], stride=1, activation_fn=act,scope='conv%d'%(i),reuse=reuse)
            local_out = current

        with tf.variable_scope('fusion',reuse=reuse):
            grid_chan_size = intensity_bin*coef
            current = act(local_out + global_out)
            A = slim.conv2d(current,grid_chan_size,[3,3], stride=1, activation_fn=None,scope='conv',reuse=reuse)

        with tf.variable_scope('guide_curve'):
            npts = 15
            nchans = 3

            idtity = np.identity(nchans, dtype=np.float32) + np.random.randn(1).astype(np.float32)*1e-4
            ccm = tf.get_variable('ccm', dtype=tf.float32, initializer=idtity)   # initializer could be np array
            ccm_bias = tf.get_variable('ccm_bias', shape=[nchans,], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            guidemap = tf.matmul(tf.reshape(input, [-1, nchans]), ccm)    #input_tensor shap should be (1,hei,wid,nchans),or will be faulty
            guidemap = tf.nn.bias_add(guidemap, ccm_bias, name='ccm_bias_add')  #bias: A 1-D Tensor with size matching the last dimension of value.
            guidemap = tf.reshape(guidemap, tf.shape(input))

            shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
            shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, np.newaxis,:]
            shifts_ = np.tile(shifts_, (1, 1, 1, nchans, 1))

            guidemap = tf.expand_dims(input, 4)   # 5
            shifts = tf.get_variable('shifts', dtype=tf.float32, initializer=shifts_)

            slopes_ = np.zeros([1, 1, 1, nchans, npts], dtype=np.float32)
            slopes_[:, :, :, :, 0] = 1.0
            slopes = tf.get_variable('slopes', dtype=tf.float32, initializer=slopes_)

            guidemap = tf.reduce_sum(slopes*tf.nn.relu(guidemap-shifts), reduction_indices=[4])

            guidemap = slim.conv2d(inputs=guidemap,num_outputs=1, kernel_size=1, weights_initializer=tf.constant_initializer(1.0/nchans),
                                    biases_initializer=tf.constant_initializer(0),activation_fn=None, reuse=reuse,scope='channel_mixing')

            guidemap = tf.clip_by_value(guidemap, 0, 1)

        with tf.variable_scope('guided_upsample'):
            out = []
            input_aug = tf.concat([input,tf.ones_like(input[:,:,:,0:1],dtype=tf.float32)],3)
            shape = tf.shape(A)
            A = tf.reshape(A,[shape[0],shape[1],shape[2],intensity_bin,coef])
            Au = layers.guided_upsampling(A,guidemap)
            for i in range(3):
                out.append(tf.reduce_sum(input_aug*Au[:,:,:,i*4:(i+1)*4],3,keepdims=True))		
            final = tf.concat(out,3)

        
           
    return final




def ResNet(input,middle=None,is_middle=False,reuse=False):
    current = input
    act = lrelu
    with tf.variable_scope('ResNet'):
        with tf.variable_scope('Restore-Net'):
            current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_init',reuse=reuse)
            for j in range(10):
                add = current
                current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_%d0'%(j),reuse=reuse)
                current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_%d1'%(j),reuse=reuse)
                current = current + add
            current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_final',reuse=reuse)
            restore = slim.conv2d(current,12,[1,1], activation_fn=None,scope='conv_subpixel',reuse=reuse)
            restore = tf.depth_to_space(restore,2)
        if is_middle==True:
            restore_gamma = tf.clip_by_value(middle,0.0,1.0)**(1.0/2.2)
        else:
            restore_gamma = tf.clip_by_value(restore,0.0,1.0)**(1.0/2.2)
    with tf.variable_scope('Enhance-Net'):
        current=slim.conv2d(restore_gamma,32,[3,3], activation_fn=act,scope='conv_init',reuse=reuse)
        for j in range(6):
            current=slim.conv2d(current,32,[3,3],rate=2**(i), activation_fn=act,scope='conv_%d0'%(j),reuse=reuse)
        current=slim.conv2d(current,32,[3,3], activation_fn=act,scope='conv_final0',reuse=reuse)
        enhance=slim.conv2d(current,3,[1,1], activation_fn=None,scope='conv_final1',reuse=reuse)
    return restore,enhance




def ResNet1(input,reuse=False):
    current = input
    act = lrelu
    with tf.variable_scope('ResNet'):
        with tf.variable_scope('Restore-Net'):
            current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_init',reuse=reuse)
            for j in range(10):
                add = current
                current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_%d0'%(j),reuse=reuse)
                current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_%d1'%(j),reuse=reuse)
                current = current + add
            current=slim.conv2d(current,64,[3,3], activation_fn=act,scope='conv_final',reuse=reuse)
            restore = slim.conv2d(current,12,[1,1], activation_fn=None,scope='conv_subpixel',reuse=reuse)
            restore = tf.depth_to_space(restore,2)

    with tf.variable_scope('Enhance-Net'):
        current=slim.conv2d(restore,32,[3,3], activation_fn=act,scope='conv_init',reuse=reuse)
        for j in range(6):
            current=slim.conv2d(current,32,[3,3],rate=2**(i), activation_fn=act,scope='conv_%d0'%(j),reuse=reuse)
        current=slim.conv2d(current,32,[3,3], activation_fn=act,scope='conv_final0',reuse=reuse)
        enhance=slim.conv2d(current,3,[1,1], activation_fn=None,scope='conv_final1',reuse=reuse)
    return enhance