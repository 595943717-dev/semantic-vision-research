import glob
import os
import csv
import numpy as np
import scipy.io
from pylsd_master.pylsd.lsd import lsd

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class nyu:

    def __init__(self, data_dir_path="./data", split='all', keep_data_in_memory=True, mat_file_path=None,
                 normalise_coordinates=False, remove_borders=True, extract_lines=False, external_lines_folder=None):
        """
        NYU-VP dataset class
        :param data_dir_path: Path where the CSV files containing VP labels etc. are stored
        :param split: train, val, test, trainval or all
        :param keep_data_in_memory: whether data shall be cached in memory
        :param mat_file_path: path to the MAT file containing the original NYUv2 dataset
        :param normalise_coordinates: normalise all point coordinates to a range of (-1,1)
        :param remove_borders: ignore the white borders around the NYU images
        :param extract_lines: do not use the pre-extracted line segments
        """
        self.keep_in_mem = keep_data_in_memory
        self.normalise_coords = normalise_coordinates
        self.remove_borders = remove_borders
        self.extract_lines = extract_lines
        self.external_lines = external_lines_folder

        self.vps_files = glob.glob(os.path.join(data_dir_path, "vps*"))
        self.lsd_line_files = glob.glob(os.path.join(data_dir_path, "lsd_lines*"))
        self.labelled_line_files = glob.glob(os.path.join(data_dir_path, "labelled_lines*"))
        self.vps_files.sort()
        self.lsd_line_files.sort()
        self.labelled_line_files.sort()

        if split == "train":
            self.set_ids = list(range(0, 1000))
        elif split == "val":
            self.set_ids = list(range(1000, 1224))
        elif split == "trainval":
            self.set_ids = list(range(0, 1224))
        elif split == "test":
            self.set_ids = list(range(1224, 1449))
        elif split == "all":
            self.set_ids = list(range(0, 1449))
        else:
            assert False, "invalid split: %s " % split

        self.dataset = [None for _ in self.set_ids]

        self.data_mat = None
        if mat_file_path is not None:
            self.data_mat = scipy.io.loadmat(mat_file_path, variable_names=["images"])

        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02

        K = np.matrix([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])

        if normalise_coordinates:
            S = np.matrix([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
            K = S * K

        self.Kinv = K.I

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        """
        Returns a sample from the dataset.
        :param key: image ID within the selected dataset split
        :return: dictionary containing vanishing points, line segments, original image
        """
        id = self.set_ids[key]

        datum = self.dataset[key]

        if datum is None:

            lsd_line_segments = None

            if self.data_mat is not None:
                image_rgb = self.data_mat['images'][:,:,:,id]
                image = rgb2gray(image_rgb)

                if self.remove_borders:
                    image_ = image[6:473,7:631].copy()
                else:
                    image_ = image

                if self.extract_lines:

                    lsd_line_segments = lsd(image_)

                    if self.remove_borders:
                        lsd_line_segments[:,0] += 7
                        lsd_line_segments[:,2] += 7
                        lsd_line_segments[:,1] += 6
                        lsd_line_segments[:,3] += 6
            else:
                image_rgb = None

            if self.external_lines:
                lines_file = os.path.join(self.external_lines, "%04d.csv" % id)

                lsd_line_segments = []
                with open(lines_file, 'r') as csv_file:
                    reader = csv.reader(csv_file, delimiter=' ')
                    for line in reader:
                        p1x = float(line[0])
                        p1y = float(line[1])
                        p2x = float(line[2])
                        p2y = float(line[3])
                        lsd_line_segments += [np.array([p1x, p1y, p2x, p2y])]
                lsd_line_segments = np.vstack(lsd_line_segments)

            if lsd_line_segments is None:
                lsd_line_segments = []
                with open(self.lsd_line_files[id], 'r') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=' ')
                    for line in reader:
                        p1x = float(line['point1_x'])
                        p1y = float(line['point1_y'])
                        p2x = float(line['point2_x'])
                        p2y = float(line['point2_y'])
                        lsd_line_segments += [np.array([p1x, p1y, p2x, p2y])]
                lsd_line_segments = np.vstack(lsd_line_segments)

            labelled_line_segments = []
            with open(self.labelled_line_files[id], 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ')
                for line in reader:
                    lines_per_vp = []
                    for i in range(1,5):
                        key_x1 = 'line%d_x1' % i
                        key_y1 = 'line%d_y1' % i
                        key_x2 = 'line%d_x2' % i
                        key_y2 = 'line%d_y2' % i

                        if line[key_x1] == '':
                            break

                        p1x = float(line[key_x1])
                        p1y = float(line[key_y1])
                        p2x = float(line[key_x2])
                        if line[key_y2] == '433q':
                            assert False, self.labelled_line_files[id]
                        p2y = float(line[key_y2])

                        ls = np.array([p1x, p1y, p2x, p2y])
                        lines_per_vp += []
                        if self.normalise_coords:
                            ls[0] -= 320
                            ls[2] -= 320
                            ls[1] -= 240
                            ls[3] -= 240
                            ls[0:4] /= 320.
                        lines_per_vp += [ls]
                    lines_per_vp = np.vstack(lines_per_vp)
                    labelled_line_segments += [lines_per_vp]

            if self.normalise_coords:
                lsd_line_segments[:,0] -= 320
                lsd_line_segments[:,2] -= 320
                lsd_line_segments[:,1] -= 240
                lsd_line_segments[:,3] -= 240
                lsd_line_segments[:,0:4] /= 320.

            line_segments = np.zeros((lsd_line_segments.shape[0], 7+2+3+3))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li,0], lsd_line_segments[li,1], 1])
                p2 = np.array([lsd_line_segments[li,2], lsd_line_segments[li,3], 1])
                centroid = 0.5*(p1+p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid

            vp_list = []
            vd_list = []
            with open(self.vps_files[id]) as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                for ri, row in enumerate(reader):
                    if ri == 0: continue
                    vp = np.array([float(row[1]), float(row[2]), 1])
                    if self.normalise_coords:
                        vp[0] -= 320
                        vp[1] -= 240
                        vp[0:2] /= 320.
                    vp_list += [vp]

                    vd = np.array(self.Kinv * np.matrix(vp).T)
                    vd /= np.linalg.norm(vd)
                    vd_list += [vd]
            vps = np.vstack(vp_list)
            vds = np.vstack(vd_list)

            datum = {'line_segments': line_segments, 'VPs': vps, 'id': id, 'VDs': vds, 'image': image_rgb,
                     'labelled_lines': labelled_line_segments}

            for vi in range(datum['VPs'].shape[0]):
                datum['VPs'][vi,:] /= np.linalg.norm(datum['VPs'][vi,:])

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum