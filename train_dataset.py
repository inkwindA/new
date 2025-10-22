import numpy
import scipy.sparse
import os
import torch.utils.data
import einops

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(TrainDataset, self).__init__()
        self.path = path
        # 获取所有文件名并排序
        self.filenames = sorted([filename for filename in os.listdir(os.path.join(path, 'ldct')) if filename.endswith('.npy')])
        
        self.ldct_list = []
        self.ndct_list = []

        for filename in self.filenames:
            ldct_path = os.path.join(path, 'ldct', filename)
            ndct_path = os.path.join(path, 'ndct', filename)
           
            ldct = self.__load_image(ldct_path)
            ndct = self.__load_image(ndct_path)

            self.ldct_list.append(ldct)
            self.ndct_list.append(ndct)

    def __getitem__(self, index):
        # 获取当前帧
        current_ldct = self.ldct_list[index]
        current_ndct = self.ndct_list[index]
        
        # 获取前一帧（如果存在，否则复制当前帧）
        if index > 0:
            prev_ldct = self.ldct_list[index - 1]
        else:
            prev_ldct = current_ldct.copy()
            
        # 获取后一帧（如果存在，否则复制当前帧）
        if index < len(self.ldct_list) - 1:
            next_ldct = self.ldct_list[index + 1]
        else:
            next_ldct = current_ldct.copy()
        
        # 将三帧合并为多帧输入 (3, height, width)
        multi_frame_ldct = numpy.concatenate([prev_ldct, current_ldct, next_ldct], axis=0)
        
        return multi_frame_ldct, current_ndct

    def __len__(self):
        return len(self.ldct_list)

    def __load_image(self, path):
        image = numpy.load(path , allow_pickle=True)
        image = image.astype('float32')

        return numpy.expand_dims(image, axis=0)
    
    def __data_augmentation(self, image, mode):
        image = einops.rearrange(image, 'c h w -> h w c')
        if mode == 0:  # original
            pass
        elif mode == 1:
            image = numpy.flipud(image)
        elif mode == 2:
            image = numpy.rot90(image)
        elif mode == 3:
            image = numpy.rot90(image)
            image = numpy.flipud(image)
        elif mode == 4:
            image = numpy.rot90(image, k=2)
        elif mode == 5:
            image = numpy.rot90(image, k=2)
            image = numpy.flipud(image)
        elif mode == 6:
            image = numpy.rot90(image, k=3)
        elif mode == 7:
            image = numpy.rot90(image, k=3)
            image = numpy.flipud(image)
        image = einops.rearrange(image, 'h w c -> c h w')
        return image.copy()
