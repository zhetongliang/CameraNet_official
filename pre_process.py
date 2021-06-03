import tensorflow as tf
import numpy as np
import cv2

def Color_space_convert(img,matrix):
    with tf.name_scope("Color_space_convert"):
        img_pixels = tf.reshape(img, [-1, 3])
        matrix = matrix.transpose()
        matrix = tf.constant(matrix)
        img_pixels = tf.matmul(img_pixels, matrix)
        img_pixels = tf.clip_by_value(img_pixels,0.0,1.0)
    return tf.reshape(img_pixels, tf.shape(img))

def decorrelation(img):
    with tf.name_scope("decorrelation"):
        img_pixels = tf.reshape(img, [-1, 3])
        matrix = np.array([
            [0.5925,    0.5821,    0.5568],
            [-0.4755,   -0.3052,    0.8251],
            [-0.6502,    0.7536,   -0.0960]])
        matrix = matrix.transpose()
        matrix = tf.constant(matrix,dtype=tf.float32)
        img_pixels = tf.matmul(img_pixels, matrix)
    return tf.reshape(img_pixels, tf.shape(img))
def correlation(img):
    with tf.name_scope("correlation"):
        img_pixels = tf.reshape(img, [-1, 3])
        matrix = np.array([
            [0.5925,   -0.4755,   -0.6502],
            [0.5821,   -0.3052,    0.7536],
            [0.5568,    0.8251,   -0.0960]])
        matrix = matrix.transpose()
        matrix = tf.constant(matrix,dtype=tf.float32)
        img_pixels = tf.matmul(img_pixels, matrix)
    return tf.reshape(img_pixels, tf.shape(img))

def linrgb_to_yuv(linrgb):
    with tf.name_scope("linrgb_to_yuv"):
        linrgb_pixels = tf.reshape(linrgb, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [0.299, -0.169, 0.5], # R
                [0.587, -0.331, -0.419], # G
                [0.114, 0.5, -0.081], # B
            ])
        linrgb_pixels = tf.matmul(linrgb_pixels, matrix)
    return tf.reshape(linrgb_pixels, tf.shape(linrgb))


def Nexus6P2XYZ(Nexus6P):
    with tf.name_scope("Nexus6P2XYZ"):
        Nexus6P_pixels = tf.reshape(Nexus6P, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [1.2064, 0.3657, -0.0359], # R
                [0.1583, 0.7536, -0.2928], # G
                [0.3926, 0.1224, 1.8871], # B
            ])
        Nexus6P_pixels = tf.matmul(Nexus6P_pixels, matrix)
    return tf.reshape(Nexus6P_pixels, tf.shape(Nexus6P))
def Nexus6P2XYZ_noise(Nexus6P):
    with tf.name_scope("Nexus6P2XYZ"):
        Nexus6P_pixels = tf.reshape(Nexus6P, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [1.2064, 0.3657, -0.0359], # R
                [0.1583, 0.7536, -0.2928], # G
                [0.3926, 0.1224, 1.8871], # B
            ])
        matrix = tf.pow(matrix,2)
        Nexus6P_pixels = tf.matmul(Nexus6P_pixels, matrix)
    return tf.reshape(Nexus6P_pixels, tf.shape(Nexus6P))


def srgb_to_lin(srgb):
    with tf.name_scope("srgb_to_lin"):
        srgb = tf.clip_by_value(srgb,0.0,1.0)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
        exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        rgb_pixels = tf.clip_by_value(rgb_pixels,0.0,1.0)
    return tf.reshape(rgb_pixels, tf.shape(srgb))


def lin_to_srgb(lin):
    with tf.name_scope("lin_to_srgb"):
        rgb_pixels = tf.clip_by_value(lin, 0.0, 1.0)
        linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
        exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask
        srgb_pixels = tf.clip_by_value(srgb_pixels,0.0,1.0)
    return tf.reshape(srgb_pixels, tf.shape(lin))


def lin_to_srgb_dif(linrgb):
    with tf.name_scope("lin_to_srgb_dif"):
        linrgb_pixels = tf.clip_by_value(linrgb, 0.0, 1.0)
        thres = 0.0031308
        changed_thres = thres*12.92
        lin = tf.minimum(linrgb_pixels,0.0031308)
        lin = lin*12.92
        expo = tf.maximum(linrgb_pixels,0.0031308)
        expo = (1.0 + 0.055)*expo**(1.0/2.4) - 0.055
    return lin + expo - changed_thres

def srgb_to_xyz(srgb):
    with tf.name_scope("srgb_to_xyz"):
        srgb = tf.clip_by_value(srgb,0.0,1.0)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
        exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
        xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)
        xyz_pixels = tf.clip_by_value(xyz_pixels,0.0,1.0)
    return tf.reshape(xyz_pixels, tf.shape(srgb))

def lin_to_xyz(rgb):
    with tf.name_scope("lin_to_xyz"):
        rgb = tf.clip_by_value(rgb,0.0,1.0)
        rgb_pixels = tf.reshape(rgb,[-1,3])
        rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
        xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)
    return tf.reshape(tf.clip_by_value(xyz_pixels,0.0,1.0), tf.shape(rgb))



def xyz_to_linRGB(xyz,clip=False):
    with tf.name_scope("xyz_to_linRGB"):
        xyz_pixels = tf.reshape(xyz, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [3.2404542, -0.9692660, 0.0556434], # R
                [-1.5371385, 1.8760108, -0.2040259], # G
                [-0.4985314, 0.0415560, 1.0572252], # B
            ])
        linRGB_pixels = tf.matmul(xyz_pixels, matrix)
        if clip is True:
            linRGB_pixels = tf.clip_by_value(linRGB_pixels,0.0,1.0)
    return tf.reshape(linRGB_pixels, tf.shape(xyz))

def xyz_to_srgb(x):
    return lin_to_srgb(xyz_to_linRGB(x))


'''
def xyz_to_sRGB_dif(xyz):
    with tf.name_scope("xyz_to_linRGB"):
        xyz_pixels = tf.reshape(xyz, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [3.2410, -0.9692, 0.0556], # R
                [-0.5374, 1.8760, -0.2040], # G
                [-0.4986, 0.0416, 1.0570], # B
            ])
        linRGB_pixels = tf.matmul(xyz_pixels, matrix)
	thres = 0.00304
	new_thres = thres*12.92
        a = 0.055
	lin = tf.minimum(linRGB_pixels,thres)
	lin = 12.92*lin
	expon = tf.maximum(linRGB_pixels,thres)
	expon = (1+a)*(expon**(1.0/2.4)) - a
        new = lin - new_thres + expon - new_thres + new_thres
    return tf.reshape(linRGB_pixels, tf.shape(xyz))
'''

def Argb_to_xyz(argb):
    with tf.name_scope("Argb_to_lab"):
        argb = tf.clip_by_value(argb,0.0,1.0)
        argb_pixels = tf.reshape(argb, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [0.5767309, 0.2973769, 0.0270343], # R
                [0.1855540, 0.6273491, 0.0706872], # G
                [0.1881852, 0.0752741, 0.9911085], # B
            ])
        xyz_pixels = tf.matmul(argb_pixels, matrix)
        xyz_pixels = tf.clip_by_value(xyz_pixels,0.0,1.0)
    return tf.reshape(xyz_pixels, tf.shape(argb))

def xyz_to_argb(xyz):
    with tf.name_scope("xyz_to_argb"):
        xyz = tf.clip_by_value(xyz,0.0,1.0)
        xyz_pixels = tf.reshape(xyz, [-1, 3])
        matrix = tf.constant([
                #    X        Y          Z
                [2.0413690, -0.9692660, 0.0134474], # R
                [-0.5649464, 1.8760108, -0.1183897], # G
                [-0.3446944, 0.0415560, 1.0154096], # B
            ])
        argb_pixels = tf.matmul(xyz_pixels, matrix)
        argb_pixels = tf.clip_by_value(argb_pixels,0.0,1.0)
    return tf.reshape(argb_pixels, tf.shape(xyz))

def xyz_to_lab_dif(xyz):
    with tf.name_scope("xyz_to_cielab_dif"):
        xyz = tf.clip_by_value(xyz, 0.0, 1.0)
        xyz_pixels = tf.reshape(xyz, [-1, 3])
        xyz_pixels_n = tf.multiply(xyz_pixels, [1.0/0.950456, 1.0, 1.0/1.088754])
        epsilon = 6.0/29.0
        ## Linear path ##
        linear = tf.minimum(xyz_pixels_n,epsilon**3)
        linear = (1.0/3.0)*((29.0/6.0)**2)*linear + 16.0/116.0 - epsilon
        ## exponential path ##
        expon = tf.maximum(xyz_pixels_n,epsilon**3)
        expon = expon**(1.0/3.0) - epsilon
        ## final ##
        final = linear + expon + epsilon

        # convert to lab
        fxfyfz_to_lab = tf.constant([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
        ])
        lab_pixels = tf.matmul(final, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

    return tf.reshape(lab_pixels, tf.shape(xyz))/500.0



def xyz_to_lab(xyz):
    with tf.name_scope("xyz_to_cielab"):
        xyz_pixels = tf.reshape(xyz, [-1, 3])
        # normalize for D65 white point
        xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
        epsilon = 6/29
        linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
        exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask
        # convert to lab
        fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
        lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
    return tf.reshape(lab_pixels, tf.shape(xyz))





#############################################################################################



def upsample_lp(x,channel,lap_filt,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        if lap_filt==None:
            filt = [[0.0025,0.0125,0.0200,0.0125,0.0025],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0200,0.1000,0.1600,0.1000,0.0200],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0025,0.0125,0.0200,0.0125,0.0025]]
            filter_mask = np.zeros(shape=[5,5,channel,channel],dtype=np.float32)
            for i in range(channel):
                filter_mask[:,:,i,i] = filt
            lap_filt = tf.constant(filter_mask,dtype=tf.float32)
        x = 4.0*x
        zero1 = tf.zeros(shape=tf.shape(x),dtype=tf.float32)
        zero2 = tf.zeros(shape=tf.shape(x),dtype=tf.float32)
        zero3 = tf.zeros(shape=tf.shape(x),dtype=tf.float32)
        contlist = []
        for i in range(channel):
            contlist.extend([x[:,:,:,i:i+1],zero1[:,:,:,i:i+1],zero2[:,:,:,i:i+1],zero3[:,:,:,i:i+1]])
        cont = tf.concat(contlist,3)
        zero = tf.depth_to_space(cont,2)
        zero_pad = tf.pad(zero,[[0,0],[2,2],[2,2],[0,0]],'SYMMETRIC')
        return tf.nn.conv2d(zero_pad,lap_filt,strides=[1,1,1,1],padding='VALID',name=scope)

def Reconstruct_laplacian_pyramid(img_pyramid,channel,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        Guassian_layers = []
        filt = [[0.0025,0.0125,0.0200,0.0125,0.0025],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0200,0.1000,0.1600,0.1000,0.0200],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0025,0.0125,0.0200,0.0125,0.0025]]
        filter_mask = np.zeros(shape=[5,5,channel,channel],dtype=np.float32)
        for i in range(channel):
            filter_mask[:,:,i,i] = filt
        lap_filt = tf.constant(filter_mask,dtype=tf.float32)
        ## Begin the reconstruction
        img_pyramid.reverse()
        base = img_pyramid[0]
        Guassian_layers.append(base)
        for i in range(1,len(img_pyramid)):        
            base_up = upsample_lp(base,channel,lap_filt,scope='upsample%d'%(i))
            base = base_up + img_pyramid[i]
            Guassian_layers.append(base)
        Guassian_layers.reverse()
    return base

def Gaussian_pyramid(img,channel,levels,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        Guassian_layers = []
        filt = [[0.0025,0.0125,0.0200,0.0125,0.0025],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0200,0.1000,0.1600,0.1000,0.0200],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0025,0.0125,0.0200,0.0125,0.0025]]
        filter_mask = np.zeros(shape=[5,5,channel,channel],dtype=np.float32)
        for i in range(channel):
            filter_mask[:,:,i,i] = filt
        lap_filt = tf.constant(filter_mask,dtype=tf.float32)
        ## Begin the construction
        current = img
        Guassian_layers.append(current)
        for i in range(levels-1):  
            ## Gaussian blurring
            current_pad = tf.pad(current,[[0,0],[2,2],[2,2],[0,0]],'SYMMETRIC')
            feature = tf.nn.conv2d(current_pad,lap_filt,strides=[1,1,1,1],padding='VALID',name='gaussian_level%d_down'%(i))
            feature = feature[:,::2,::2,:]
            Guassian_layers.append(feature)
            current = feature
    return Guassian_layers




def Laplacian_pyramid(img,channel,levels,scope,reuse=False):
    with tf.variable_scope('U_Net_Res_reuse',reuse=reuse):
        filt = [[0.0025,0.0125,0.0200,0.0125,0.0025],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0200,0.1000,0.1600,0.1000,0.0200],
                [0.0125,0.0625,0.1000,0.0625,0.0125],
                [0.0025,0.0125,0.0200,0.0125,0.0025]]
        filter_mask = np.zeros(shape=[5,5,channel,channel],dtype=np.float32)
        for i in range(channel):
            filter_mask[:,:,i,i] = filt
        lap_filt = tf.constant(filter_mask,dtype=tf.float32)
        ## convolution
        collections = []
        current = img
        for i in range(levels):
            ## Gaussian blurring
            current_pad = tf.pad(current,[[0,0],[2,2],[2,2],[0,0]],'SYMMETRIC')
            feature = tf.nn.conv2d(current_pad,lap_filt,strides=[1,1,1,1],padding='VALID',name='lap_level%d_down'%(i))
            feature = feature[:,::2,::2,:]
            ## Upsample and compute residual
            upsample = upsample_lp(feature,channel,lap_filt,'lap_level%d_up'%(i),reuse)
            residual = current - upsample
            collections.append(residual)
            current = feature
        collections.append(feature)
    return collections


def nor(x):
    max_x = x.max()
    min_x = x.min()
    return (x-min_x)/(max_x-min_x)


def Gaussian_smoothing(x,kernel_size,sigma,chan,name):
    with tf.name_scope("Gaussian_smoothing"):
        kx = cv2.getGaussianKernel(kernel_size,sigma)
        ky = cv2.getGaussianKernel(kernel_size,sigma)
        kernel = np.multiply(kx,np.transpose(ky)) 
        filter_mask = np.zeros(shape=[kernel_size,kernel_size,chan,chan],dtype=np.float32)
        for i in range(chan):
            filter_mask[:,:,i,i] = kernel
        filter_tensor = tf.constant(filter_mask,dtype=tf.float32)
        pad_size = (kernel_size-1)/2
        x_pad = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]],'SYMMETRIC')
        return tf.nn.conv2d(x_pad,filter_tensor,strides=[1,1,1,1],padding='VALID',name=name)
        

def moment_feature(x,channel):
    with tf.name_scope("feature_extraction"):
        filt = cv2.getGaussianKernel(3,1)
        filt = np.multiply(filt,np.transpose(filt)) 
        filter_mask = np.zeros(shape=[3,3,3,3],dtype=np.float32)
        for i in range(channel):
            filter_mask[:,:,i,i] = filt
        filter_tensor = tf.constant(filter_mask,dtype=tf.float32)
        ## Filtering the input to remove noise
        x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        x = tf.nn.conv2d(x,filter_tensor,strides=[1,1,1,1],padding='VALID',name='PreFilter')
        ## Gray world
        norm = 1.0
        gray_world = (tf.reduce_mean(x**(norm),[1,2]))**(1/norm)
        ## Gray edge
        norm = 1.0
        x_p = tf.pad(x,[[0,0],[1,0],[1,0],[0,0]],'SYMMETRIC')
        Dx = x[:,:-1,:-1,:] - x[:,1:,:-1,:]
        Dy = x[:,:-1,1:,:] - x[:,:-1,:-1,:]
        Dxy = tf.sqrt(Dx**2 + Dy**2)
        gray_edge = (tf.reduce_mean(Dxy**(norm),[1,2]))**(1/norm)	
        ## max-RGB
        max_RGB = tf.reduce_max(x,[1,2])
        ## Fusion
        fuse = tf.concat([gray_world,gray_edge,max_RGB],1)
    return fuse



def chroma_histogram(x,num_bin=10):
    with tf.name_scope("histogram_feature"): 
        num_bin_float = np.float32(num_bin)
        num_bin_int = np.int32(num_bin)
        shape = tf.shape(x)
        size = shape[1]*shape[2]
        luma = tf.reduce_sum(x,3,keepdims=True)
        R = tf.div(x[:,:,:,0:1],luma+1e-5)
        G = tf.div(x[:,:,:,1:2],luma+1e-5)
        R = tf.cast(R*num_bin_float,dtype=tf.int32)
        G = tf.cast(G*num_bin_float,dtype=tf.int32)
        feature = 0
        for i in range(num_bin_int):   
            feature = feature + (i*num_bin_int)*tf.cast(tf.equal(R,i),dtype=tf.int32)
        for j in range(num_bin_int):	        
            feature = feature + j*tf.cast(tf.equal(G,j),dtype=tf.int32)
        histogram = tf.bincount(feature,minlength=num_bin_int**2,maxlength=num_bin_int**2)   # So, only batch_size=1 is support
        histogram = tf.reshape(histogram,[num_bin_int,num_bin_int])
        histogram_1d = histogram[0,:]
        for i in range(1,num_bin_int):
            histogram_1d = tf.concat([histogram_1d,histogram[i,0:num_bin_int-i]],0)
        histogram_1d = tf.expand_dims(histogram_1d,0)
        histogram_1d = tf.cast(histogram_1d,dtype=tf.float32)/tf.cast(size,dtype=tf.float32)
    return histogram_1d




'''
#######Tesing code
import os
import cv2
level = 5
channel = 3
path = '/home/justin/Desktop/disk/HDR/Nexus_6P/version_1/Train/gt'
files = os.listdir(path)
files.sort()
outpath = '/home/justin/Desktop/disk/HDR/Nexus_6P/version_1/Train/test'
img_tensor = tf.placeholder(tf.float32,shape=[1,512,512,channel])
img_pyramid = Laplacian_pyramid(img_tensor,channel,level,'pyramid')
reconstruction = Reconstruct_laplacian_pyramid(img_pyramid,channel,'reconstruct')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20):
    img_file = path+'/'+files[i]
    img = cv2.imread(img_file)        
    img = np.float32(img)/255.0
   # img = np.sum(img,2,keepdims=True)/3.0 
    img = np.expand_dims(img,0)
    patch = img[:,0:512,0:512,:]
    img_pyramid_,reconstruction_ = sess.run([img_pyramid,reconstruction],feed_dict={img_tensor:patch})
    for ii in range(level+1):
	img_out = nor(np.abs(img_pyramid_[ii]))
        img_out = img_out[0,:,:,:]
        cv2.imwrite(outpath+'/img%02d_%02d.png'%(i,ii),np.uint8(img_out*255.0))
    final = np.concatenate([reconstruction_,patch],2)
    cv2.imwrite(outpath+'/reconstruction_%02d.png'%(i),np.uint8(final[0,:,:,:]*255.0))
sess.close()
'''


