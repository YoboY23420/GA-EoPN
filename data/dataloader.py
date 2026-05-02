import glob
import torch
import nibabel as nib
from torch.utils import data
from itertools import permutations
import pickle

class torch_Dataset_OASIS(data.Dataset):
    def __init__(self, img_dir, seg_dir, mode):
        super(torch_Dataset_OASIS, self).__init__()
        self.img = glob.glob(img_dir + '*.nii.gz')
        self.img.sort(key=lambda x: int(x[66:-7]))
        self.seg = glob.glob(seg_dir + '*.nii.gz')
        self.seg.sort(key=lambda x: int(x[66:-7]))
        assert len(self.img) == len(self.seg), 'Image number != Segmentation number'
        print('len(self.img) = {}, len(self.seg) = {}'.format(len(self.img), len(self.seg)))
        self.mode = mode
        self.training_img_pair = list(permutations(self.img[0:255], 2))
        self.training_seg_pair = list(permutations(self.seg[0:255], 2))
        self.testing_img_pair = list((moving, atlas) for moving in self.img[256:401] for atlas in self.img[401:405])
        self.testing_seg_pair = list((moving, atlas) for moving in self.seg[256:401] for atlas in self.seg[401:405])
    def __len__(self):
        if self.mode == 'train':
            assert len(self.training_img_pair) == len(self.training_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.training_img_pair)
        elif self.mode == 'test':
            assert len(self.testing_img_pair) == len(self.testing_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.testing_img_pair)
    def __getitem__(self, item):
        if self.mode == 'train':
            mi = torch.from_numpy(nib.load(self.training_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            fi = torch.from_numpy(nib.load(self.training_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            ml = torch.from_numpy(nib.load(self.training_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
            fl = torch.from_numpy(nib.load(self.training_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
            pair = (self.training_img_pair[item][0][66:-7], self.training_img_pair[item][1][66:-7])
            return pair, mi.float(), fi.float(), ml.float(), fl.float()
        elif self.mode == 'test':
            mi = torch.from_numpy(nib.load(self.testing_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            fi = torch.from_numpy(nib.load(self.testing_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            ml = torch.from_numpy(nib.load(self.testing_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
            fl = torch.from_numpy(nib.load(self.testing_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
            pair = (self.testing_img_pair[item][0][66:-7], self.testing_img_pair[item][1][66:-7])
            return pair, mi.float(), fi.float(), ml.float(), fl.float()

def torch_Dataloader_OASIS(img_dir, seg_dir, mode, batch_size, random_seed=None):
    Dataset_OASIS = torch_Dataset_OASIS(img_dir, seg_dir, mode)
    # 这里shuffle设置成了false，因为网上说已经有batch_size了，就不需要shuffle来进行随机了，将shuffle设置为FALSE即可，
    # 但是我看来网上有人又可以将其设置为True，但是是batch_size=4的时候
    if random_seed is None:
        loader = data.DataLoader(dataset=Dataset_OASIS, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(random_seed)
        '''或者也可以这样写'''
        # torch.manual_seed(random_seed)
        # g = torch.Generator()
        loader = data.DataLoader(dataset=Dataset_OASIS, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False, generator=g)
    return loader

class torch_Dataset_LPBA40(data.Dataset):
    def __init__(self, img_dir, seg_dir, mode):
        super(torch_Dataset_LPBA40, self).__init__()
        self.img = glob.glob(img_dir + '*.nii.gz')
        self.img.sort(key=lambda x: int(x[60:-7]))
        self.seg = glob.glob(seg_dir + '*.nii.gz')
        self.seg.sort(key=lambda x: int(x[60:-7]))
        assert len(self.img) == len(self.seg), 'Image number != Segmentation number'
        print('len(self.img) = {}, len(self.seg) = {}'.format(len(self.img), len(self.seg)))
        self.mode = mode
        self.training_img_pair = list(permutations(self.img[0:28], 2))
        self.training_seg_pair = list(permutations(self.seg[0:28], 2))
        self.testing_img_pair = list(permutations(self.img[28:38], 2))
        self.testing_seg_pair = list(permutations(self.seg[28:38], 2))
    def __len__(self):
        if self.mode == 'train':
            assert len(self.training_img_pair) == len(self.training_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.training_img_pair)
        elif self.mode == 'test':
            assert len(self.testing_img_pair) == len(self.testing_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.testing_img_pair)
    def __getitem__(self, item):
        if self.mode == 'train':
            mi = torch.from_numpy(nib.load(self.training_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            fi = torch.from_numpy(nib.load(self.training_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            ml = torch.from_numpy(nib.load(self.training_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
            fl = torch.from_numpy(nib.load(self.training_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
            pair = (self.training_img_pair[item][0][60:-7], self.training_img_pair[item][1][60:-7])
            return pair, mi.float(), fi.float(), ml.float(), fl.float()
        elif self.mode == 'test':
            mi = torch.from_numpy(nib.load(self.testing_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            fi = torch.from_numpy(nib.load(self.testing_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            ml = torch.from_numpy(nib.load(self.testing_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
            fl = torch.from_numpy(nib.load(self.testing_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
            pair = (self.testing_img_pair[item][0][60:-7], self.testing_img_pair[item][1][60:-7])
            return pair, mi.float(), fi.float(), ml.float(), fl.float()
        else:
            return None

def torch_Dataloader_LPBA40(img_dir, seg_dir, mode, batch_size):
    Dataset_LPBA40 = torch_Dataset_LPBA40(img_dir, seg_dir, mode)
    loader = data.DataLoader(dataset=Dataset_LPBA40, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    return loader

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
class torch_Dataset_IXI(data.Dataset):
    def __init__(self, tra_dir, val_dir, mode):
        super(torch_Dataset_IXI, self).__init__()
        self.tra = glob.glob(tra_dir + '*.pkl')
        self.tra.sort(key=lambda x: int(x[76:-4]))
        self.val = glob.glob(val_dir + '*.pkl')
        self.val.sort(key=lambda x: int(x[75:-4]))
        print('len(self.tra) = {}, len(self.val) = {}'.format(len(self.tra), len(self.val)))
        self.mode = mode
        self.training_tra_pair = list(permutations(self.tra, 2))
        self.testing_val_pair = list((moving, atlas) for moving in self.val[5:115] for atlas in self.val[0:5])
    def __len__(self):
        if self.mode == 'train':
            return len(self.training_tra_pair)
        elif self.mode == 'test':
            return len(self.testing_val_pair)
    def __getitem__(self, item):
        if self.mode == 'train':
            mi, ml = pkload(self.training_tra_pair[item][0])
            fi, fl = pkload(self.training_tra_pair[item][1])
            mi = torch.from_numpy(mi)
            ml = torch.from_numpy(ml)
            fi = torch.from_numpy(fi)
            fl = torch.from_numpy(fl)
            pair = (self.training_tra_pair[item][0][76:-4], self.training_tra_pair[item][1][76:-4])
            return pair, mi.float(), fi.float(), ml.float(), fl.float()
        elif self.mode == 'test':
            mi, ml = pkload(self.testing_val_pair[item][0])
            fi, fl = pkload(self.testing_val_pair[item][1])
            mi = torch.from_numpy(mi)
            ml = torch.from_numpy(ml)
            fi = torch.from_numpy(fi)
            fl = torch.from_numpy(fl)
            pair = (self.testing_val_pair[item][0][75:-4], self.testing_val_pair[item][1][75:-4])
            return pair, mi.float(), fi.float(), ml.float(), fl.float()


def torch_Dataloader_IXI(tra_dir, val_dir, mode, batch_size, random_seed=None):
    Dataset_IXI = torch_Dataset_IXI(tra_dir, val_dir, mode)
    # 这里shuffle设置成了false，因为网上说已经有batch_size了，就不需要shuffle来进行随机了，将shuffle设置为FALSE即可，
    # 但是我看来网上有人又可以将其设置为True，但是是batch_size=4的时候
    if random_seed is None:
        loader = data.DataLoader(dataset=Dataset_IXI, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(random_seed)
        '''或者也可以这样写'''
        # torch.manual_seed(random_seed)
        # g = torch.Generator()
        loader = data.DataLoader(dataset=Dataset_IXI, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False, generator=g)
    return loader

# if __name__ == '__main__':
#     #将数据统一缩小一半，存放进服务器的新文件夹中
#     OASIS_img_dir = '/Extra/yzy/Medical_Image_Registration/3D_brain_MRI/affine_img_ordered/'
#     OASIS_seg_dir = '/Extra/yzy/Medical_Image_Registration/3D_brain_MRI/affine_seg_ordered/'
#     LPBA40_img_dir = '/Extra/yzy/Medical_Image_Registration/3D_LPBA40/lpba40_affine_img_ordered/'
#     LPBA40_seg_dir = '/Extra/yzy/Medical_Image_Registration/3D_LPBA40/lpba40_affine_seg_ordered/'
#
#     train_dataset_OASIS = torch_Dataset_OASIS(OASIS_img_dir, OASIS_seg_dir, 'train', resized_shape=(80, 96, 112), resize=False)
#     test_dataset_OASIS = torch_Dataset_OASIS(OASIS_img_dir, OASIS_seg_dir, 'test', resized_shape=(80, 96, 112), resize=False)
#     test_dataset_LPBA40 = torch_Dataset_LPBA40(LPBA40_img_dir, LPBA40_seg_dir, 'test', resized_shape=(80, 96, 112), resize=False)
#     train_dataloader_OASIS = torch_Dataloader_OASIS(OASIS_img_dir, OASIS_seg_dir, 'train', 1, (80, 96, 112), False, None)
#     test_dataloader_OASIS = torch_Dataloader_OASIS(OASIS_img_dir, OASIS_seg_dir, 'test', 1, (80, 96, 112), False, None)
#     test_dataloader_LPBA40 = torch_Dataloader_LPBA40(LPBA40_img_dir, LPBA40_seg_dir, 'test', 1, (80, 96, 112), False)
#
#     # 将图像和分割缩小size，以保证能喂的进GPU中装得下，同时保留图像信息
#     mi_affine = nib.load(OASIS_img_dir+'img_1.nii.gz').affine
#     ml_affine = nib.load(OASIS_seg_dir+'seg_1.nii.gz').affine
#     mi = nib.load(OASIS_img_dir + 'img_1.nii.gz').get_fdata()[48:-48, 31:-33, 3:-29]
#     ml = nib.load(OASIS_seg_dir + 'seg_1.nii.gz').get_fdata()[48:-48, 31:-33, 3:-29]
#     mi_shape = mi.shape
#     # resized_shape = (80, 96, 112)
#     resized_shape = (160, 192, 224)
#     scale_factor = [resized_shape[0] / mi_shape[0], resized_shape[1] / mi_shape[1], resized_shape[2] / mi_shape[2]]
#     resized_mi = zoom(mi, scale_factor, order=1) / 255.0
#     resized_ml = zoom(ml, scale_factor, order=0)
#     nib.save(nib.Nifti1Image(resized_mi, mi_affine), "/home/ubuntu/yzy/Medical Image Registration/Third_paper/XMorpher_reproduce/image.nii.gz")
#     nib.save(nib.Nifti1Image(resized_ml, ml_affine), "/home/ubuntu/yzy/Medical Image Registration/Third_paper/XMorpher_reproduce/segmentation.nii.gz")