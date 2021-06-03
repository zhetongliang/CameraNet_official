import tensorflow as tf
import os

########################## Input Pipeline #####################
def augment_data(data,ps=768,nchan=6):
    with tf.name_scope('Data_augmentation'):
        shape = tf.shape(data)
        data = tf.image.random_flip_left_right(data)
        data = tf.image.random_flip_up_down(data)   
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32)
        data = tf.case([(tf.equal(angle, 1), lambda: tf.image.rot90(data, k=1)),
                         (tf.equal(angle, 2), lambda: tf.image.rot90(data, k=2)),
                         (tf.equal(angle, 3), lambda: tf.image.rot90(data, k=3))],
                        lambda: data)

        data.set_shape([None, None, nchan])    
        data = tf.random_crop(data, tf.stack([ps, ps, nchan]))
        data.set_shape([None, None, nchan])  
    '''
        factor = 3/4
    scale = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if scale==0:
            data0 = tf.image.resize_images(data[:,:,0:3],[ps*factor,ps*factor])
            data1 = tf.image.resize_images(data[:,:,3:6],[ps*factor,ps*factor])
            data2 = tf.image.resize_images(data[:,:,6:9],[ps*factor,ps*factor])
            data = tf.concat([data0,data1,data2],2)
        '''
    return data

def file_input_pipeline_stage1(input_filelist,middle_filelist,input_dir,middle_dir,batchsize=1,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.xreadlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])   
    input_data = tf.decode_raw(input_file, tf.float32)
    hei_in,wid_in,input_img = tf.cast(input_data[0],dtype=tf.int32)/2,tf.cast(input_data[1],dtype=tf.int32)/2,tf.cast(input_data[2:],dtype=tf.float32)
    input_img = tf.reshape(input_img,[hei_in,wid_in,4])

    middle_file = tf.read_file(input_queue[1])   
    middle_data = tf.decode_raw(middle_file, tf.float32)
    hei_middle,wid_middle,middle_img = tf.cast(middle_data[0],dtype=tf.int32),tf.cast(middle_data[1],dtype=tf.int32),tf.cast(middle_data[2:],dtype=tf.float32)
    middle_img = tf.reshape(middle_img,[hei_middle,wid_middle,4])

    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1)
        elif angle==1:
           input_img,middle_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2)
        elif angle==2:
           input_img,middle_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3)
        else:
           input_img,middle_img = input_img,middle_img
      
    inout = tf.random_crop(tf.concat([input_img,middle_img],2), tf.stack([patchsize, patchsize, 8]))
    input_patch,middle_patch = inout[:,:,0:4],inout[:,:,4:]
    
    input_patch_, middle_patch_ = tf.train.batch(tensors=[input_patch,middle_patch],batch_size=batchsize,num_threads=num_threads)

    return input_patch_, middle_patch_

def file_input_pipeline_stage2(input_filelist,middle_filelist,input_dir,middle_dir,batchsize=1,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.xreadlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])   
    input_data = tf.decode_raw(input_file, tf.float32)
    hei_in,wid_in,input_img = tf.cast(input_data[0],dtype=tf.int32),tf.cast(input_data[1],dtype=tf.int32),tf.cast(input_data[2:],dtype=tf.float32)
    input_img = tf.reshape(input_img,[hei_in,wid_in,4])

    middle_file = tf.read_file(input_queue[1])
    middle_file = tf.image.decode_png(middle_file,dtype=tf.uint16, channels=3)
    middle_img = tf.to_float(middle_file)/65535.0

    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1)
        elif angle==1:
           input_img,middle_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2)
        elif angle==2:
           input_img,middle_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3)
        else:
           input_img,middle_img = input_img,middle_img
            
    crop_hei = tf.random_uniform((), minval=0, maxval=hei_in-patchsize/2, dtype=tf.int32)
    crop_wid = tf.random_uniform((), minval=0, maxval=wid_in-patchsize/2, dtype=tf.int32)
    input_patch = tf.image.crop_to_bounding_box(input_img,crop_hei,crop_wid,patchsize/2,patchsize/2)
    middle_patch = tf.image.crop_to_bounding_box(middle_img,crop_hei*2,crop_wid*2,patchsize,patchsize)
    input_patch_, middle_patch_ = tf.train.batch(tensors=[input_patch,middle_patch],batch_size=batchsize,num_threads=num_threads)

    return input_patch_, middle_patch_

def file_input_pipeline_stage3(input_filelist,middle_filelist,input_dir,middle_dir,batchsize=1,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.xreadlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])
    input_file = tf.image.decode_png(input_file, dtype=tf.uint16,channels=3)
    input_img = tf.to_float(input_file)/65535.0

    middle_file = tf.read_file(input_queue[1])
    middle_file = tf.image.decode_jpeg(middle_file, channels=3)
    middle_img = tf.to_float(middle_file)/255.0

    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1)
        elif angle==1:
           input_img,middle_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2)
        elif angle==2:
           input_img,middle_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3)
        else:
           input_img,middle_img = input_img,middle_img
            
    inout = tf.random_crop(tf.concat([input_img,middle_img],2), tf.stack([patchsize, patchsize, 6]))
    input_patch,middle_patch = inout[:,:,0:3],inout[:,:,3:]

    input_patch_, middle_patch_ = tf.train.batch(tensors=[input_patch,middle_patch],batch_size=batchsize,num_threads=num_threads)

    return input_patch_, middle_patch_
def file_input_pipeline_stage4(input_filelist,middle_filelist,input_dir,middle_dir,batchsize=1,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.xreadlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])
    input_file = tf.image.decode_jpeg(input_file, channels=3)
    input_img = tf.to_float(input_file)/255.0

    middle_file = tf.read_file(input_queue[1])
    middle_file = tf.image.decode_jpeg(middle_file, channels=3)
    middle_img = tf.to_float(middle_file)/255.0

    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1)
        elif angle==1:
           input_img,middle_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2)
        elif angle==2:
           input_img,middle_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3)
        else:
           input_img,middle_img = input_img,middle_img
        
    inout = tf.random_crop(tf.concat([input_img,middle_img],2), tf.stack([patchsize, patchsize, 6]))
    input_patch,middle_patch = inout[:,:,0:3],inout[:,:,3:]

    input_patch_, middle_patch_ = tf.train.batch(tensors=[input_patch,middle_patch],batch_size=batchsize,num_threads=num_threads)

    return input_patch_, middle_patch_

def file_input_pipeline_noise(input_filelist,middle_filelist,output_filelist,param_filelist,input_dir,middle_dir,output_dir,param_dir,batchsize=1,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(output_filelist, 'r') as fid:
        output_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(param_filelist, 'r') as fid:
        param_filelist_ = [l.strip() for l in fid.xreadlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    output_filelist__ = [os.path.join(output_dir, f) for f in output_filelist_]
    param_filelist__ = [os.path.join(param_dir, f) for f in param_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__,output_filelist__,param_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])   
    input_data = tf.decode_raw(input_file, tf.float32)
    hei,wid,input_img = tf.cast(input_data[0],dtype=tf.int32)/2,tf.cast(input_data[1],dtype=tf.int32)/2,tf.cast(input_data[2:],dtype=tf.float32)
    input_img = tf.reshape(input_img,[hei,wid,4])

    middle_file = tf.read_file(input_queue[1])
    middle_file = tf.image.decode_png(middle_file, dtype=tf.uint16, channels=3)
    middle_img = tf.to_float(middle_file)/65535.0

    output_file = tf.read_file(input_queue[2])
    output_file = tf.image.decode_jpeg(output_file, channels=3)
    output_img = tf.to_float(output_file)/255.0

    param_file = tf.read_file(input_queue[3])   
    param_data = tf.decode_raw(param_file, tf.float32)
    param_data = tf.cast(param_data,dtype=tf.float32)
    param_data = tf.reshape(param_data,[6,])

    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img,output_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img),tf.image.flip_left_right(output_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img,output_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img),tf.image.flip_up_down(output_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1),tf.image.rot90(output_img, k=1)
        elif angle==1:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2),tf.image.rot90(output_img, k=2)
        elif angle==2:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3),tf.image.rot90(output_img, k=3)
        else:
           input_img,middle_img,output_img = input_img,middle_img,output_img
            
    crop_hei = tf.random_uniform((), minval=0, maxval=hei-patchsize/2, dtype=tf.int32)
    crop_wid = tf.random_uniform((), minval=0, maxval=wid-patchsize/2, dtype=tf.int32)
    input_patch = tf.image.crop_to_bounding_box(input_img,crop_hei,crop_wid,patchsize/2,patchsize/2)
    middle_patch = tf.image.crop_to_bounding_box(middle_img,crop_hei*2,crop_wid*2,patchsize,patchsize)
    output_patch = tf.image.crop_to_bounding_box(output_img,crop_hei*2,crop_wid*2,patchsize,patchsize)


    input_patch_, middle_patch_,output_patch_,param_data_ = tf.train.batch(tensors=[input_patch,middle_patch,output_patch,param_data],batch_size=batchsize,num_threads=num_threads)

    return input_patch_, middle_patch_,output_patch_,param_data_

def file_input_pipeline(input_filelist,middle_filelist,output_filelist,input_dir,middle_dir,output_dir,batchsize=1,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.readlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.readlines()]
    with open(output_filelist, 'r') as fid:
        output_filelist_ = [l.strip() for l in fid.readlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    output_filelist__ = [os.path.join(output_dir, f) for f in output_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__,output_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])   
    input_data = tf.decode_raw(input_file, tf.float32)
    hei,wid,input_img = tf.cast(input_data[0],dtype=tf.int32)//2,tf.cast(input_data[1],dtype=tf.int32)//2,tf.cast(input_data[2:],dtype=tf.float32)
    input_img = tf.reshape(input_img,[hei,wid,4])

    middle_file = tf.read_file(input_queue[1])
    middle_file = tf.image.decode_png(middle_file, dtype=tf.uint16, channels=3)
    middle_img = tf.to_float(middle_file)/65535.0

    output_file = tf.read_file(input_queue[2])
    output_file = tf.image.decode_jpeg(output_file, channels=3)
    output_img = tf.to_float(output_file)/255.0
    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img,output_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img),tf.image.flip_left_right(output_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img,output_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img),tf.image.flip_up_down(output_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1),tf.image.rot90(output_img, k=1)
        elif angle==1:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2),tf.image.rot90(output_img, k=2)
        elif angle==2:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3),tf.image.rot90(output_img, k=3)
        else:
           input_img,middle_img,output_img = input_img,middle_img,output_img
            
    crop_hei = tf.random_uniform((), minval=0, maxval=hei-patchsize//2, dtype=tf.int32)
    crop_wid = tf.random_uniform((), minval=0, maxval=wid-patchsize//2, dtype=tf.int32)
    input_patch = tf.image.crop_to_bounding_box(input_img,crop_hei,crop_wid,patchsize//2,patchsize//2)
    middle_patch = tf.image.crop_to_bounding_box(middle_img,crop_hei*2,crop_wid*2,patchsize,patchsize)
    output_patch = tf.image.crop_to_bounding_box(output_img,crop_hei*2,crop_wid*2,patchsize,patchsize)


    input_patch_, middle_patch_,output_patch_ = tf.train.batch(tensors=[input_patch,middle_patch,output_patch],batch_size=batchsize,num_threads=num_threads)

    return input_patch_, middle_patch_,output_patch_


def file_input_pipeline_LSD(input_filelist,middle_filelist,output_filelist,input_dir,middle_dir,output_dir,
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.readlines()]
    with open(middle_filelist, 'r') as fid:
        middle_filelist_ = [l.strip() for l in fid.readlines()]
    with open(output_filelist, 'r') as fid:
        output_filelist_ = [l.strip() for l in fid.readlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    middle_filelist__ = [os.path.join(middle_dir, f) for f in middle_filelist_]
    output_filelist__ = [os.path.join(output_dir, f) for f in output_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,middle_filelist__,output_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])   
    input_data = tf.decode_raw(input_file, tf.float32)
    hei,wid,input_img = tf.cast(input_data[0],dtype=tf.int32),tf.cast(input_data[1],dtype=tf.int32),tf.cast(input_data[2:],dtype=tf.float32)
    input_img = tf.reshape(input_img,[hei,wid,4])

    middle_file = tf.read_file(input_queue[1])
    middle_file = tf.image.decode_png(middle_file, dtype=tf.uint16, channels=3)
    middle_img = tf.to_float(middle_file)/65535.0

    output_file = tf.read_file(input_queue[2])
    output_file = tf.image.decode_jpeg(output_file, channels=3)
    output_img = tf.to_float(output_file)/255.0
    
    if is_augment==True:
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img,output_img = tf.image.flip_left_right(input_img),tf.image.flip_left_right(middle_img),tf.image.flip_left_right(output_img)
        is_true = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        if is_true==0:
            input_img,middle_img,output_img = tf.image.flip_up_down(input_img),tf.image.flip_up_down(middle_img),tf.image.flip_up_down(output_img)
        angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32) 
        if angle==0:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=1),tf.image.rot90(middle_img, k=1),tf.image.rot90(output_img, k=1)
        elif angle==1:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=2),tf.image.rot90(middle_img, k=2),tf.image.rot90(output_img, k=2)
        elif angle==2:
           input_img,middle_img,output_img = tf.image.rot90(input_img, k=3),tf.image.rot90(middle_img, k=3),tf.image.rot90(output_img, k=3)
        else:
           input_img,middle_img,output_img = input_img,middle_img,output_img
            
    crop_hei = tf.random_uniform((), minval=0, maxval=hei-patchsize//2, dtype=tf.int32)
    crop_wid = tf.random_uniform((), minval=0, maxval=wid-patchsize//2, dtype=tf.int32)
    input_patch = tf.image.crop_to_bounding_box(input_img,crop_hei,crop_wid,patchsize//2,patchsize//2)
    middle_patch = tf.image.crop_to_bounding_box(middle_img,crop_hei*2,crop_wid*2,patchsize,patchsize)
    output_patch = tf.image.crop_to_bounding_box(output_img,crop_hei*2,crop_wid*2,patchsize,patchsize)


    input_patch_, middle_patch_,output_patch_ = tf.train.batch(tensors=[input_patch,middle_patch,output_patch],batch_size=1,num_threads=num_threads)

    return input_patch_, middle_patch_,output_patch_



def file_input_pipeline_FiveK(input_filelist,merged_filelist,output_filelist,input_dir,merged_dir,output_dir, 
                                                              num_epochs=1000,patchsize=768,is_augment=True,is_shuffle=True,num_threads=1):
    with open(input_filelist, 'r') as fid:
        input_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(merged_filelist, 'r') as fid:
        merged_filelist_ = [l.strip() for l in fid.xreadlines()]
    with open(output_filelist, 'r') as fid:
        output_filelist_ = [l.strip() for l in fid.xreadlines()]
    
    input_filelist__ = [os.path.join(input_dir, f) for f in input_filelist_]
    merged_filelist__ = [os.path.join(merged_dir, f) for f in merged_filelist_]
    output_filelist__ = [os.path.join(output_dir, f) for f in output_filelist_]
    
    input_queue = tf.train.slice_input_producer(tensor_list=[input_filelist__,merged_filelist__,output_filelist__],shuffle=is_shuffle,
                              num_epochs=num_epochs,capacity=32)

    input_file = tf.read_file(input_queue[0])
    input_file = tf.image.decode_png(input_file, dtype=tf.uint16, channels=3)
    input_img = tf.to_float(input_file)/65535.0

    merged_file = tf.read_file(input_queue[1])
    merged_file = tf.image.decode_png(merged_file, dtype=tf.uint16, channels=3)
    merged_img = tf.to_float(merged_file)/65535.0

    output_file = tf.read_file(input_queue[2])
    output_file = tf.image.decode_jpeg(output_file, channels=3)
    output_img = tf.to_float(output_file)/255.0
    

    inout = tf.concat([input_img,merged_img,output_img],2)
    if is_augment==True:
        inout = augment_data(inout,ps=patchsize,nchan=9)
    #in_patch,other_patch_t = inout[:,:,0:3],inout[:,:,3:]
    #in_patch,noise_t = Add_gaussian(in_patch,noise_t)  
    #param_t = tf.concat([noise_t,others_t],0) 
    #inout = tf.concat([in_patch,other_patch_t],2)
    elif is_augment==False:
        inout = tf.random_crop(inout, tf.stack([patchsize, patchsize, 9]))
    image,merged,label = inout[:,:,0:3],inout[:,:,3:6],inout[:,:,6:]
    image_,merged_,label_ = tf.train.batch(tensors=[image,merged,label],batch_size=1,num_threads=num_threads)

    return image_,merged_,label_


