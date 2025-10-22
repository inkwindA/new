import os
import torch
import numpy


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(TestDataset, self).__init__()
        self.ldct_filename_list = []
        self.ndct_filename_list = []
        # 获取所有文件名并排序
        filenames = sorted([filename for filename in os.listdir(os.path.join(path, 'ldct'))])
        for filename in filenames:
            self.ldct_filename_list.append(os.path.join(path, 'ldct', filename))
            self.ndct_filename_list.append(os.path.join(path, 'ndct', filename))
        
        # 预加载所有图像
        self.ldct_images = []
        self.ndct_images = []
        
        for ldct_path, ndct_path in zip(self.ldct_filename_list, self.ndct_filename_list):
            ldct_image, _, _ = self.__load_image(ldct_path)
            ndct_image, _, _ = self.__load_image(ndct_path)
            self.ldct_images.append(ldct_image)
            self.ndct_images.append(ndct_image)

    def __getitem__(self, index):
        # 获取当前帧
        current_ldct = self.ldct_images[index]
        current_ndct = self.ndct_images[index]
        
        # 获取前一帧（如果存在，否则复制当前帧）
        if index > 0:
            prev_ldct = self.ldct_images[index - 1]
        else:
            prev_ldct = current_ldct.clone()
            
        # 获取后一帧（如果存在，否则复制当前帧）
        if index < len(self.ldct_images) - 1:
            next_ldct = self.ldct_images[index + 1]
        else:
            next_ldct = current_ldct.clone()
        
        # 将三帧合并为多帧输入 (3, height, width)
        multi_frame_ldct = torch.cat([prev_ldct, current_ldct, next_ldct], dim=0)
        
        return current_ndct, multi_frame_ldct

    def __len__(self):
        return len(self.ldct_filename_list)

    def __load_image(self, path):
        image = numpy.load(path)
        image = image.astype('float32')
        mean = numpy.mean(image)
        var = numpy.var(image)

        return torch.from_numpy(numpy.expand_dims(image, axis=0)), mean, var
