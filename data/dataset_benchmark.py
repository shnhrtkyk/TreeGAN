from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np

class BenchmarkDataset(data.Dataset):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None):
        self.npoints = npoints
        self.root = root
        self.catfile = './data/synsetoffset2category.txt'
        self.cat = {}
        self.uniform = uniform
        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
                
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_sampling = os.path.join(self.root, self.cat[item], 'sampling')

            fns = sorted(os.listdir(dir_point))

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'), os.path.join(dir_sampling, token + '.sam')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        if self.uniform:
            choice = np.loadtxt(fn[3]).astype(np.int64)
            assert len(choice) == self.npoints, "Need to match number of choice(2048) with number of vertices."
        else:
            choice = np.random.randint(0, len(seg), size=self.npoints)

        point_set = point_set[choice]
        seg = seg[choice]

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class MyDataset(data.Dataset):
    def __init__(self, root, npoints=8192, utransform=None):
        self.npoints = npoints
        self.root = root
        self.imglist = []
        self.pointlist = []
        self.imgpath = root + "/img/"
        self.pointpath = root + "/point/"
        self.img_list = files = glob.glob(self.imgpath + "/*.tif")
        self.point_list = files = glob.glob(self.pointpath + "/*.txt")

        for file in self.point_list:
            print(file)
            # point cloud 取得
            src = np.loadtxt(file)
            normlized_xyz = np.zeros((npoints, 3))
            self.coord_min, self.coord_max = np.amin(src, axis=0)[:3], np.amax(src, axis=0)[:3]
            src[:, 0] = src[:, 0] / self.coord_max[0]
            src[:, 1] = src[:, 1] / self.coord_max[1]
            src[:, 2] = src[:, 2] / self.coord_max[2]
            if(len(src) >=npoints):
                np.random.shuffle(src)
                normlized_xyz[:,:]=src[:,:npoints]
            else:
                normlized_xyz[:,:len(src)]=src[:,:]

            self.pointlist.append(src)

        for file in self.img_list:
            print(file)
            # geotiff取得
            src = rasterio.open(file)
            arr = src.read()  # read all raster values
            arr = arr.astype(np.uint8)
            self.imglist.append(arr)
                

        self.data_num = len(self.img_list)
        

    def __getitem__(self, index):
        img = self.imglist[index]
        point = self.pointlist[index]
        point = torch.from_numpy(point)
        if self.transform:
            img = self.transform(img)


        return point_set, img

    def __len__(self):
        return len(self.imglist)


