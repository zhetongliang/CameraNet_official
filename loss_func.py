import tensorflow as tf
import numpy as np
import pre_process as pp

def feature_histogram(x,num_bin=10,bin_with=0.1,bin_center=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],
                      num_chan=3, is_global_pool=True):
    ## Stage 0 filter
    stage0_filter = np.zeros(shape=[1,1,num_chan,num_bin*num_chan],dtype=np.float32)
    for i in range(num_chan):
        stage0_filter[0,0,i,i*num_bin:(i+1)*num_bin] = 1.0
    filt = tf.constant(value=stage0_filter,shape=[1,1,num_chan,num_bin*num_chan],name='stage0_unitfil')
    current = tf.nn.conv2d(input=x,filter=filt,strides=[1,1,1,1],padding='SAME',name='stage0_unitconv')
    
    ## Stage 0 bias
    stage0_bias = np.zeros(shape=[1,1,1,num_bin*num_chan],dtype=np.float32)
    for i in range(num_chan):
        stage0_bias[0,0,0,i*num_bin:(i+1)*num_bin] = bin_center
    add = tf.constant(value=stage0_bias,shape=[1,1,1,num_bin*num_chan],name='stage0_bias')    
    current = tf.subtract(current,add) 
    
    ## The rest
    current = tf.abs(current)    
    current = tf.multiply(current,-bin_with)
    current = tf.add(current,1.0)
    current = tf.nn.relu(current)
    if is_global_pool:
        final = tf.reduce_mean(current,3,keep_dims=True)
    else:
        final = current
        
    return final

def multi_l1(label,output):
    loss = 0.0
    num_scale = len(output)
    for i in range(num_scale):
        shape = tf.shape(output[i])
        label_scale = tf.image.resize_images(label,[shape[1],shape[2]])
        loss += tf.reduce_mean(tf.abs(label_scale - output[i]))
    return loss/tf.cast(num_scale,tf.float32) 

def multi_l1_log(label,output):
    loss = 0.0
    num_scale = len(output)
    for i in range(num_scale):
        current_tensor = output[i]
        shape = tf.shape(current_tensor)
        label_scale = tf.image.resize_images(label,[shape[1],shape[2]])
        current_tensor_c = tf.maximum(current_tensor,0.0)
        loss += tf.reduce_mean(tf.abs(tf.log(label_scale+0.001) - tf.log(current_tensor_c+0.001)))
    return loss/tf.cast(num_scale,tf.float32) 


def multi_l1_lab(label,output):
    loss = 0.0
    num_scale = len(output)
    for i in range(num_scale):
        shape = tf.shape(output[i])
        label_scale = tf.image.resize_images(label,[shape[1],shape[2]])
        loss += tf.reduce_mean(tf.abs(pp.xyz_to_lab_dif(label_scale) - pp.xyz_to_lab_dif(output[i])))
    return loss/tf.cast(num_scale,tf.float32) 
  
def multi_l1_list(label,output):
    loss = 0.0
    num_scale = len(output)
    for i in range(num_scale):
        loss += tf.reduce_mean(tf.abs(label[i] - output[i]))
    return loss/tf.cast(num_scale,tf.float32)  

def multi_l1_list_std(label,output,start):
    loss = 0.0
    num_scale = len(output)
    label_current = label
    for i in range(start,start+num_scale):
        label_current = tf.space_to_depth(label_current,2**(i))
        loss += tf.reduce_mean(tf.abs(label_current - output[i-start]))
    return loss/tf.cast(num_scale,tf.float32)   


def tri_training(loss0,variable_scope0,lr0,w0,loss1,variable_scope1,lr1,w1,loss2,variable_scope2,lr2,w2,global_lr):
    varlist0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=variable_scope0)
    varlist1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=variable_scope1) 
    varlist2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=variable_scope2)
    
    loss = loss0*w0 + loss1*w1 + loss2*w2
    
    opt0 = tf.train.AdamOptimizer(learning_rate=global_lr*lr0).minimize(loss,var_list=varlist0)
    opt1 = tf.train.AdamOptimizer(learning_rate=global_lr*lr1).minimize(loss,var_list=varlist1)
    opt2 = tf.train.AdamOptimizer(learning_rate=global_lr*lr2).minimize(loss,var_list=varlist2)
    train_op = tf.group(opt0,opt1,opt2)

    '''
    opt0 = tf.train.AdamOptimizer(learning_rate=global_lr*lr0) 
    opt1 = tf.train.AdamOptimizer(learning_rate=global_lr*lr1)
    opt2 = tf.train.AdamOptimizer(learning_rate=global_lr*lr2)


    grads = tf.gradients(loss,varlist0+varlist1+varlist2)
    grad0,grad1,grad2 = grads[:len(varlist0)],grads[len(varlist0):len(varlist1)],grads[len(varlist0)+len(varlist1):]

    train_op0 = opt0.apply_gradients(zip(grad0,varlist0))
    train_op1 = opt1.apply_gradients(zip(grad1,varlist1))
    train_op2 = opt2.apply_gradients(zip(grad2,varlist2))
    train_op = tf.group(train_op0,train_op1,train_op2)
    '''
    return train_op

def angular_loss(vec1,vec2):
    safe_v = 0.999999
    vec_normal1 = tf.nn.l2_normalize(vec1,1)
    vec_normal2 = tf.nn.l2_normalize(vec2,1)
    dot = tf.reduce_sum(vec_normal1*vec_normal2)
    dot = tf.clip_by_value(dot,-safe_v,safe_v)
    angle = tf.acos(dot)/3.14159
    return angle


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 - 1:size//2 + 1, -size//2 - 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1.0  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2.0*mu1_mu2 + C1)*(2.0*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2.0*mu1_mu2 + C1)*(2.0*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    eps = 1e-8
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    mssim = tf.maximum(mssim,eps)
    mcs = tf.maximum(mcs,eps)
    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value





