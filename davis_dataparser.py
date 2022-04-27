import os
import glob
from collections import namedtuple

ListDataOpt = namedtuple("ListDataPath", ["id", "flow_image", "semantic_label"])


class Davis_DataParser(object):
    def __init__(self, data_root, resolution, category):
        # self.is_train = is_train
        # self.image_list = self.get_consecutive_image_path(data_root=data_root, dtype=dtype)
        self.flow_list = self.get_flow_path(data_root=data_root, resolution=resolution, category=category)
        self.semantic_label_path = self.get_label_path(data_root=data_root, resolution=resolution, category=category)
        self.data_path_list = []
        for i in range(len(self.semantic_label_path)):
            item = ListDataOpt(i, self.flow_list[i], self.semantic_label_path[i])
            self.data_path_list.append(item)

    def get_flow_path(self, data_root, resolution="480p", category="blackswan"):
        """
        generate consecutive image pair in each scene
        :param data_root: root directory of Sintel dataset
        :return: directory for flow images
        """
        image_dir = os.path.join(data_root, "JPEGImages", resolution, category)
        flow_img_path = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        return flow_img_path

    def get_label_path(self, data_root, resolution="480p", category="blackswan"):
        """

        :param data_root: root directory of Sintel dataset
        :param mode: select from training, testing
        :return:
        """
        image_dir = os.path.join(data_root, "Annotations", resolution, category)
        label_path = glob.glob(os.path.join(image_dir, '*.png'))
        return label_path


if __name__ == '__main__':
    # dataparser = SintelDataParser(data_root="/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/MPI-Sintel-complete",
    #                               dtype="clean")
    # print(len(dataparser.data_path_list))
    dataparser = Davis_DataParser(data_root="/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/DAVIS-data/DAVIS",
                                   resolution="480p", category="blackswan")
    res = dataparser.get_label_path(data_root="/media/zlu6/4caa1062-1ae5-4a99-9354-0800d8a1121d/DAVIS-data/DAVIS",
                                   resolution="480p", category="blackswan")
    print(dataparser.data_path_list)
