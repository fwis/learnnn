import numpy as np
from imgaug import augmenters as iaa


chance = 1./6
seq = iaa.Sequential([
        iaa.Sometimes(chance,iaa.Fliplr(1)), # 水平方向翻转
        iaa.Sometimes(chance,iaa.Flipud(1)), # 垂直方向翻转
        iaa.Sometimes(chance*4, iaa.OneOf([iaa.Affine(rotate=0),
                                         iaa.Affine(rotate=90),
                                         iaa.Affine(rotate=180),
                                         iaa.Affine(rotate=270) # 旋转(90,180,270)度
                                         ]))],
                    random_order=True # do all of the above in random order
                    )

def patch_batch_generator(Y, label, batch_size=64, patch_width=64, patch_height=64, random_shuffle=True, augment=True):
    '''
    生成数据(data,label)

    Input:
        data_list, random_shuffle, batch_size 

    Output:
        (X_batch, Y_batch)
    '''
    
    seq_fixed = seq.to_deterministic()
    
    N = len(Y)
    # X是输出的去马赛克图像，Y是输入
    # 删除label最后一个通道（保存用）


    if random_shuffle:
        index = np.random.permutation(N)
    else:
        index = np.arange(N)
    
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        # 如果最后一个batch不够长，改变他的长度
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        Y_batch = np.zeros((current_batch_size, patch_width, patch_height, Y.shape[-1]))

#        X_batch = np.zeros((current_batch_size, patch_width, patch_height, 4))
        label_batch = np.zeros((current_batch_size, patch_width, patch_height, label.shape[-1]))
        

        for i in range(current_index, current_index + current_batch_size):
            # 把Y加入batch
            Y_batch[i - current_index] = Y[index[i]]
#            X_batch[i - current_index] = X[index[i]]
            label_batch[i - current_index] = label[index[i]]
        
        if augment:
            Y_batch = seq_fixed.augment_images(Y_batch)
#            X_batch = seq_fixed.augment_images(X_batch)
            label_batch = seq_fixed.augment_images(label_batch)
                
#        if normalize:
#            X_left_batch = X_left_batch.astype(np.float64)
#            X_left_batch = preprocess_input(X_left_batch)
        

        yield (Y_batch, label_batch)


