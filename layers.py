import tensorflow as tf
import numpy as np

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat_c(x1, x2, output_channels, in_channels, scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        pool_size = 2
        deconv_filter = tf.get_variable(shape= [pool_size, pool_size, output_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.02),name='dcf')
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

        deconv_output =  tf.concat([deconv, x2],3)
   #     deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def affine_mapping(x,in_chan,out_chan,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        mapping=tf.get_variable(shape=[in_chan+1,out_chan],initializer=tf.truncated_normal_initializer(0.0,1.0),dtype=tf.float32,name='mapping')
        x_pixels = tf.reshape(x, [-1, in_chan])
        bias = tf.ones_like(x_pixels[:,0:1])
        x_pixels = tf.concat([x_pixels,bias],1)
        x_pixels = tf.matmul(x_pixels, mapping)
        shape = tf.shape(x)
    return tf.reshape(x_pixels, [shape[0],shape[1],shape[2],out_chan])

def coeff_estimate(x,chan,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        shape = tf.shape(x)
        conv0 = slim.conv2d(current,chan,[3,3], activation_fn=lrelu,scope='g_conv0',reuse=reuse)
        pool0=slim.max_pool2d(conv0, [2, 2], padding='SAME',scope='pool0')
        conv1 = slim.conv2d(pool0,chan,[3,3], activation_fn=lrelu,scope='g_conv1',reuse=reuse)
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME',scope='pool1')
        conv2 = slim.conv2d(pool1,chan,[3,3], activation_fn=lrelu,scope='g_conv2',reuse=reuse)
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME',scope='pool2')

        conv3 = slim.conv2d(pool3,chan,[3,3], activation_fn=lrelu,scope='conv0',reuse=reuse)
        conv3 = slim.conv2d(conv3,chan,[3,3], activation_fn=lrelu,scope='conv1',reuse=reuse)

        dconv2 =  upsample_and_concat_c( conv3, conv2, chan, chan, scope='uac2',reuse=reuse )	
        dconv2 = slim.conv2d(dconv2,chan,[3,3], activation_fn=lrelu,scope='d_conv2',reuse=reuse)
        dconv1 =  upsample_and_concat_c( dconv2, conv1, chan, chan, scope='uac1',reuse=reuse )	
        dconv1 = slim.conv2d(dconv1,chan,[3,3], activation_fn=lrelu,scope='d_conv1',reuse=reuse)
        dconv0 =  upsample_and_concat_c( dconv1, conv0, chan, chan, scope='uac0',reuse=reuse )	
        dconv0 = slim.conv2d(dconv0,chan,[3,3], activation_fn=lrelu,scope='d_conv0',reuse=reuse)
    return dconv0
'''
def perpixel_conv(fp,coef,chan,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
	padding = 1
	paddings = [[0,0],[padding,padding],[padding,padding],[0,0]]
	fp = tf.pad(fp,paddings,name='pad')
	result0 = fp[:,0:-2,0:-2,:]*coef[:,:,:,:chan]
	result1 = fp[:,1:-1,0:-2,:]*coef[:,:,:,chan:chan*2]
	result2 = fp[:,2:,0:-2,:]*coef[:,:,:,chan*2:chan*3]
	result3 = fp[:,0:-2,1:-1,:]*coef[:,:,:,chan*3:chan*4]
	result4 = fp[:,1:-1,1:-1,:]*coef[:,:,:,chan*4:chan*5]
	result5 = fp[:,2:,1:-1,:]*coef[:,:,:,chan*5:chan*6]
	result6 = fp[:,0:-2,2:,:]*coef[:,:,:,chan*6:chan*7]
	result7 = fp[:,1:-1,2:,:]*coef[:,:,:,chan*7:chan*8]
	result8 = fp[:,2:,2:,:]*coef[:,:,:,chan*8:chan*9]
	result = result0 + result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + coef[:,:,:,chan*9:chan*10]

    return lrelu(result)
'''
def perpixel_affine(fp,coef,chan,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        padding = 1
        paddings = [[0,0],[padding,padding],[padding,padding],[0,0]]
        fp = tf.pad(fp,paddings,name='pad')
        result0 = fp[:,0:-2,0:-2,:]*coef[:,:,:,:chan]
        result1 = fp[:,1:-1,0:-2,:]*coef[:,:,:,chan:chan*2]
        result2 = fp[:,2:,0:-2,:]*coef[:,:,:,chan*2:chan*3]
        result3 = fp[:,0:-2,1:-1,:]*coef[:,:,:,chan*3:chan*4]
        result4 = fp[:,1:-1,1:-1,:]*coef[:,:,:,chan*4:chan*5]
        result5 = fp[:,2:,1:-1,:]*coef[:,:,:,chan*5:chan*6]
        result6 = fp[:,0:-2,2:,:]*coef[:,:,:,chan*6:chan*7]
        result7 = fp[:,1:-1,2:,:]*coef[:,:,:,chan*7:chan*8]
        result8 = fp[:,2:,2:,:]*coef[:,:,:,chan*8:chan*9]
        result = result0 + result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + coef[:,:,:,chan*9:chan*10]

    return lrelu(result)



def guided_upsampling(input_ftmp,guide_ftmp):
    # input_ftmp must be a [Batch,H,W,Intensity,Channel] shaped feature map
    # guide_ftmp must be a [Batch,H*factor,W*factor,1] shaped feature map
    def get_pixel_value(img, x, y, z):
        ## Getting parameters
        batch_size = tf.shape(img)[0]
        height = tf.shape(x)[0]
        width = tf.shape(x)[1]
        ## Preprocessing
        x = tf.cast(x,dtype=tf.int32)
        y = tf.cast(y,dtype=tf.int32)
        z = tf.cast(z,dtype=tf.int32)
        x = tf.expand_dims(x,0)
        y = tf.expand_dims(y,0)
        z = tf.expand_dims(z,0)
        x = tf.tile(x,[batch_size,1,1])
        y = tf.tile(y,[batch_size,1,1]) # x,y,z = [b,h,w]
        z = tf.tile(z,[batch_size,1,1])
        # Then b
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size,1,1))
        b = tf.tile(batch_idx, (1, height, width)) # b = [b,h,w]
    
        indices = tf.stack([b, y, x, z], 3)   
    return tf.gather_nd(img, indices)
    
    ##### Do the job
    shape = tf.shape(input_ftmp)
    height = shape[1]
    width = shape[2]
    intensity = shape[3]
    height_s,width_s,intensity_s = tf.cast(height,dtype=tf.float32),tf.cast(width,dtype=tf.float32),tf.cast(intensity,dtype=tf.float32)
    new_shape = tf.shape(guide_ftmp)
    new_height = new_shape[1]
    new_width = new_shape[2]
    # create meshgrid
    x = tf.linspace(0.0, 1.0, new_width)     
    y = tf.linspace(0.0, 1.0, new_height)
    x_t, y_t = tf.meshgrid(x, y)
    z_t = guide_ftmp[0,:,:,0]
    # Transform the coords
    x_te = x_t*(width_s-1.0)
    y_te = y_t*(height_s-1.0)
    z_te = z_t*(intensity_s-1.0)
    # 8 neighborhood
    x0 = tf.floor(x_te)
    x1 = x0 + 1.0
    y0 = tf.floor(y_te)
    y1 = y0 + 1.0
    z0 = tf.floor(z_te)
    z1 = z0 + 1.0
    x0 = tf.clip_by_value(x0, 0.0, width_s-1.0)
    x1 = tf.clip_by_value(x1, 0.0, width_s-1.0)
    y0 = tf.clip_by_value(y0, 0.0, height_s-1.0)
    y1 = tf.clip_by_value(y1, 0.0, height_s-1.0)
    z0 = tf.clip_by_value(z0, 0.0, intensity_s-1.0)
    z1 = tf.clip_by_value(z1, 0.0, intensity_s-1.0)
    Ia = get_pixel_value(input_ftmp, x0, y0,z0)
    Ib = get_pixel_value(input_ftmp, x0, y0,z1)
    Ic = get_pixel_value(input_ftmp, x1, y0,z0)
    Id = get_pixel_value(input_ftmp, x1, y0,z1)
    Ie = get_pixel_value(input_ftmp, x0, y1,z0)
    If = get_pixel_value(input_ftmp, x0, y1,z1)
    Ig = get_pixel_value(input_ftmp, x1, y1,z0)
    Ih = get_pixel_value(input_ftmp, x1, y1,z1)
    wa = tf.maximum(1.0-tf.abs(x0-x_te),0.0) * tf.maximum(1.0-tf.abs(y0-y_te),0.0) * tf.maximum(1.0-tf.abs(z0-z_te),0.0)
    wb = tf.maximum(1.0-tf.abs(x0-x_te),0.0) * tf.maximum(1.0-tf.abs(y0-y_te),0.0) * tf.maximum(1.0-tf.abs(z1-z_te),0.0)
    wc = tf.maximum(1.0-tf.abs(x1-x_te),0.0) * tf.maximum(1.0-tf.abs(y0-y_te),0.0) * tf.maximum(1.0-tf.abs(z0-z_te),0.0)
    wd = tf.maximum(1.0-tf.abs(x1-x_te),0.0) * tf.maximum(1.0-tf.abs(y0-y_te),0.0) * tf.maximum(1.0-tf.abs(z1-z_te),0.0)
    we = tf.maximum(1.0-tf.abs(x0-x_te),0.0) * tf.maximum(1.0-tf.abs(y1-y_te),0.0) * tf.maximum(1.0-tf.abs(z0-z_te),0.0)
    wf = tf.maximum(1.0-tf.abs(x0-x_te),0.0) * tf.maximum(1.0-tf.abs(y1-y_te),0.0) * tf.maximum(1.0-tf.abs(z1-z_te),0.0)
    wg = tf.maximum(1.0-tf.abs(x1-x_te),0.0) * tf.maximum(1.0-tf.abs(y1-y_te),0.0) * tf.maximum(1.0-tf.abs(z0-z_te),0.0)
    wh = tf.maximum(1.0-tf.abs(x1-x_te),0.0) * tf.maximum(1.0-tf.abs(y1-y_te),0.0) * tf.maximum(1.0-tf.abs(z1-z_te),0.0)
    wa = tf.expand_dims(tf.expand_dims(wa, axis=0),3)
    wb = tf.expand_dims(tf.expand_dims(wb, axis=0),3)
    wc = tf.expand_dims(tf.expand_dims(wc, axis=0),3)
    wd = tf.expand_dims(tf.expand_dims(wd, axis=0),3)
    we = tf.expand_dims(tf.expand_dims(we, axis=0),3)
    wf = tf.expand_dims(tf.expand_dims(wf, axis=0),3)
    wg = tf.expand_dims(tf.expand_dims(wg, axis=0),3)
    wh = tf.expand_dims(tf.expand_dims(wh, axis=0),3)
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih])
    return out


def gaussian_func(x1,x2,sigma):
    return tf.exp(-1.0*((x1-x2)**2.0)/(2.0*(sigma**2.0)))

def bilateral_joint_upsampling(input_ftmp,guide_ftmp,factor_g=0.2,factor_s=1.0,scope=None,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        shape = tf.shape(input_ftmp)
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        height_s,width_s = tf.cast(height,dtype=tf.float32),tf.cast(width,dtype=tf.float32)

        new_shape = tf.shape(guide_ftmp)
        new_height = new_shape[1]
        new_width = new_shape[2]
        new_height_s,new_width_s = tf.cast(new_height,dtype=tf.float32),tf.cast(new_width,dtype=tf.float32)
        x = tf.linspace(0.0, 1.0, new_width)     
        y = tf.linspace(0.0, 1.0, new_height)
        xt, yt = tf.meshgrid(x, y)
        xt = tf.tile(tf.expand_dims(tf.expand_dims(xt,0),3),[batchsize,1,1,1])
        yt = tf.tile(tf.expand_dims(tf.expand_dims(yt,0),3),[batchsize,1,1,1])
    
        ## Spatial 
        xd = tf.clip_by_value((width_s-1.0)*xt, 0.0, width_s-1.0)
        yd = tf.clip_by_value((height_s-1.0)*yt, 0.0, height_s-1.0)
        xd0 = tf.floor(xd)
        xd1 = xd0 + 1.0
        yd0 = tf.floor(yd)
        yd1 = yd0 + 1.0
        xd0 = tf.clip_by_value(xd0,0.0, width_s-1.0)
        xd1 = tf.clip_by_value(xd1,0.0, width_s-1.0)
        yd0 = tf.clip_by_value(yd0,0.0, height_s-1.0)
        yd1 = tf.clip_by_value(yd1,0.0, height_s-1.0)
        batch_idx = tf.range(0, batchsize)
        batch_idx = tf.reshape(batch_idx, (batchsize,1,1,1))
        bd = tf.tile(batch_idx, (1, new_height, new_width,1))
    
        indices00 = tf.concat([bd,tf.cast(yd0,tf.int32),tf.cast(xd0,tf.int32)],3)
        indices01 = tf.concat([bd,tf.cast(yd0,tf.int32),tf.cast(xd1,tf.int32)],3)
        indices10 = tf.concat([bd,tf.cast(yd1,tf.int32),tf.cast(xd0,tf.int32)],3)
        indices11 = tf.concat([bd,tf.cast(yd1,tf.int32),tf.cast(xd1,tf.int32)],3)
    
        I00 = tf.gather_nd(input_ftmp, indices00)
        I01 = tf.gather_nd(input_ftmp, indices01)
        I10 = tf.gather_nd(input_ftmp, indices10)
        I11 = tf.gather_nd(input_ftmp, indices11)
    
        #ws00 = gaussian_func(yd0,yd,factor_s) * gaussian_func(xd0,xd,factor_s)
        #ws01 = gaussian_func(yd0,yd,factor_s) * gaussian_func(xd1,xd,factor_s) 
        #ws10 = gaussian_func(yd1,yd,factor_s) * gaussian_func(xd0,xd,factor_s)
        #ws11 = gaussian_func(yd1,yd,factor_s) * gaussian_func(xd1,xd,factor_s)

        ws00 = tf.maximum(1.0-factor_s*tf.abs(yd0-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd0-xd),0.0)
        ws01 = tf.maximum(1.0-factor_s*tf.abs(yd0-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd1-xd),0.0)
        ws10 = tf.maximum(1.0-factor_s*tf.abs(yd1-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd0-xd),0.0)
        ws11 = tf.maximum(1.0-factor_s*tf.abs(yd1-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd1-xd),0.0)
    
        ## Guide
        xu = tf.clip_by_value((new_width_s-1.0)*xt, 0.0, new_width_s-1.0)
        yu = tf.clip_by_value((new_height_s-1.0)*yt, 0.0, new_height_s-1.0)
        xu0 = tf.clip_by_value((new_width_s-1.0)*xd0/(width_s-1.0), 0.0, new_width_s-1.0)
        xu1 = tf.clip_by_value((new_width_s-1.0)*xd1/(width_s-1.0), 0.0, new_width_s-1.0)
        yu0 = tf.clip_by_value((new_height_s-1.0)*yd0/(height_s-1.0), 0.0, new_height_s-1.0)
        yu1 = tf.clip_by_value((new_height_s-1.0)*yd1/(height_s-1.0), 0.0, new_height_s-1.0)
        bu = tf.tile(batch_idx, (1, new_height, new_width,1))
    
        indices00 = tf.concat([bu,tf.cast(yu0,tf.int32),tf.cast(xu0,tf.int32)],3)
        indices01 = tf.concat([bu,tf.cast(yu0,tf.int32),tf.cast(xu1,tf.int32)],3)
        indices10 = tf.concat([bu,tf.cast(yu1,tf.int32),tf.cast(xu0,tf.int32)],3)
        indices11 = tf.concat([bu,tf.cast(yu1,tf.int32),tf.cast(xu1,tf.int32)],3)
        indicestt = tf.concat([bu,tf.cast(yu,tf.int32),tf.cast(xu,tf.int32)],3)
    
        guide00 = tf.gather_nd(guide_ftmp, indices00)
        guide01 = tf.gather_nd(guide_ftmp, indices01)
        guide10 = tf.gather_nd(guide_ftmp, indices10)
        guide11 = tf.gather_nd(guide_ftmp, indices11) 
        guidett = tf.gather_nd(guide_ftmp, indicestt) 
    
	#factor_g=tf.get_variable(shape=[],initializer=tf.constant_initializer(factor_g),dtype=tf.float32,name='factor_g')
        wg00 = gaussian_func(guide00,guidett,factor_g)  
        wg01 = gaussian_func(guide01,guidett,factor_g)   
        wg10 = gaussian_func(guide10,guidett,factor_g)  
        wg11 = gaussian_func(guide11,guidett,factor_g)
    
        ## Final mearged
        weight00 = ws00*wg00
        weight01 = ws01*wg01
        weight10 = ws10*wg10
        weight11 = ws11*wg11
        weight_sum = weight00 + weight01 + weight10 + weight11 + 0.01
        I00 = I00*weight00
        I01 = I01*weight01
        I10 = I10*weight10
        I11 = I11*weight11


    return tf.add_n([I00,I01,I10,I11])/weight_sum


def bilateral_joint_upsampling_lin(input_ftmp,guide_ftmp,factor_g=5.0,factor_s=1.0,scope=None,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        shape = tf.shape(input_ftmp)
        batchsize = shape[0]
        height = shape[1]
        width = shape[2]
        height_s,width_s = tf.cast(height,dtype=tf.float32),tf.cast(width,dtype=tf.float32)

        new_shape = tf.shape(guide_ftmp)
        new_height = new_shape[1]
        new_width = new_shape[2]
        new_height_s,new_width_s = tf.cast(new_height,dtype=tf.float32),tf.cast(new_width,dtype=tf.float32)
        x = tf.linspace(0.0, 1.0, new_width)     
        y = tf.linspace(0.0, 1.0, new_height)
        xt, yt = tf.meshgrid(x, y)
        xt = tf.tile(tf.expand_dims(tf.expand_dims(xt,0),3),[batchsize,1,1,1])
        yt = tf.tile(tf.expand_dims(tf.expand_dims(yt,0),3),[batchsize,1,1,1])
    
        ## Spatial 
        xd = tf.clip_by_value((width_s-1.0)*xt, 0.0, width_s-1.0)
        yd = tf.clip_by_value((height_s-1.0)*yt, 0.0, height_s-1.0)
        xd0 = tf.floor(xd)
        xd1 = xd0 + 1.0
        yd0 = tf.floor(yd)
        yd1 = yd0 + 1.0
        xd0 = tf.clip_by_value(xd0,0.0, width_s-1.0)
        xd1 = tf.clip_by_value(xd1,0.0, width_s-1.0)
        yd0 = tf.clip_by_value(yd0,0.0, height_s-1.0)
        yd1 = tf.clip_by_value(yd1,0.0, height_s-1.0)
        batch_idx = tf.range(0, batchsize)
        batch_idx = tf.reshape(batch_idx, (batchsize,1,1,1))
        bd = tf.tile(batch_idx, (1, new_height, new_width,1))
    
        indices00 = tf.concat([bd,tf.cast(yd0,tf.int32),tf.cast(xd0,tf.int32)],3)
        indices01 = tf.concat([bd,tf.cast(yd0,tf.int32),tf.cast(xd1,tf.int32)],3)
        indices10 = tf.concat([bd,tf.cast(yd1,tf.int32),tf.cast(xd0,tf.int32)],3)
        indices11 = tf.concat([bd,tf.cast(yd1,tf.int32),tf.cast(xd1,tf.int32)],3)
    
        I00 = tf.gather_nd(input_ftmp, indices00)
        I01 = tf.gather_nd(input_ftmp, indices01)
        I10 = tf.gather_nd(input_ftmp, indices10)
        I11 = tf.gather_nd(input_ftmp, indices11)
    
        ws00 = tf.maximum(1.0-factor_s*tf.abs(yd0-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd0-xd),0.0)
        ws01 = tf.maximum(1.0-factor_s*tf.abs(yd0-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd1-xd),0.0)
        ws10 = tf.maximum(1.0-factor_s*tf.abs(yd1-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd0-xd),0.0)
        ws11 = tf.maximum(1.0-factor_s*tf.abs(yd1-yd),0.0)*tf.maximum(1.0-factor_s*tf.abs(xd1-xd),0.0)
    
        ## Guide
        xu = tf.clip_by_value((new_width_s-1.0)*xt, 0.0, new_width_s-1.0)
        yu = tf.clip_by_value((new_height_s-1.0)*yt, 0.0, new_height_s-1.0)
        xu0 = tf.clip_by_value((new_width_s-1.0)*xd0/(width_s-1.0), 0.0, new_width_s-1.0)
        xu1 = tf.clip_by_value((new_width_s-1.0)*xd1/(width_s-1.0), 0.0, new_width_s-1.0)
        yu0 = tf.clip_by_value((new_height_s-1.0)*yd0/(height_s-1.0), 0.0, new_height_s-1.0)
        yu1 = tf.clip_by_value((new_height_s-1.0)*yd1/(height_s-1.0), 0.0, new_height_s-1.0)
        bu = tf.tile(batch_idx, (1, new_height, new_width,1))
    
        indices00 = tf.concat([bu,tf.cast(yu0,tf.int32),tf.cast(xu0,tf.int32)],3)
        indices01 = tf.concat([bu,tf.cast(yu0,tf.int32),tf.cast(xu1,tf.int32)],3)
        indices10 = tf.concat([bu,tf.cast(yu1,tf.int32),tf.cast(xu0,tf.int32)],3)
        indices11 = tf.concat([bu,tf.cast(yu1,tf.int32),tf.cast(xu1,tf.int32)],3)
        indicestt = tf.concat([bu,tf.cast(yu,tf.int32),tf.cast(xu,tf.int32)],3)
    
        guide00 = tf.gather_nd(guide_ftmp, indices00)
        guide01 = tf.gather_nd(guide_ftmp, indices01)
        guide10 = tf.gather_nd(guide_ftmp, indices10)
        guide11 = tf.gather_nd(guide_ftmp, indices11) 
        guidett = tf.gather_nd(guide_ftmp, indicestt) 
    
	#factor_g=tf.get_variable(shape=[],initializer=tf.constant_initializer(factor_g),dtype=tf.float32,name='factor_g')
        wg00 = tf.maximum(1.0-factor_g*tf.abs(guide00-guidett),0.0)
        wg01 = tf.maximum(1.0-factor_g*tf.abs(guide01-guidett),0.0)
        wg10 = tf.maximum(1.0-factor_g*tf.abs(guide10-guidett),0.0)
        wg11 = tf.maximum(1.0-factor_g*tf.abs(guide11-guidett),0.0)
    
        ## Final mearged
        weight00 = ws00*wg00
        weight01 = ws01*wg01
        weight10 = ws10*wg10
        weight11 = ws11*wg11
        I00 = I00*weight00
        I01 = I01*weight01
        I10 = I10*weight10
        I11 = I11*weight11
        weight_sum = weight00 + weight01 + weight10 + weight11 + 0.0001


    return tf.add_n([I00,I01,I10,I11])/weight_sum

def spatial_conv(x,coef):
    x_pad = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
    inner0 = x_pad[:,0:-2,1:-1,0:1] * coef[:,:,:,0:1]
    inner1 = x_pad[:,1:-1,0:-2,0:1] * coef[:,:,:,1:2]
    inner2 = x_pad[:,1:-1,1:-1,0:1] * coef[:,:,:,2:3]
    inner3 = x_pad[:,1:-1,2:,0:1] * coef[:,:,:,3:4]
    inner4 = x_pad[:,2:,1:-1,0:1] * coef[:,:,:,4:5]
    return inner0+inner1+inner2+inner3+inner4
    







