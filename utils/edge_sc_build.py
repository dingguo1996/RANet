import os
import json
import numpy as np
import edge_utils
from PIL import Image
from multiprocessing import cpu_count
import threading
import os.path as osp

ignore_label = 255

id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                  7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                  18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                  28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


def id2trainId(label, reverse=False):
    label_copy = label.copy()
    if reverse:
        for v, k in id_to_trainid.items():
            label_copy[label == k] = v
    else:
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
    return label_copy


def single_core_build_edge(cpu_index, cpu_num, dataset, seg_root, edge_root, num_classes, json_label):
    image_begin = (len(json_label) // cpu_num) * cpu_index
    if cpu_index != cpu_num - 1:
        image_end = (len(json_label) // cpu_num) * (cpu_index+1)
    else:
        image_end = len(json_label)
    for img_entry in json_label[image_begin:image_end]:
        image_path, label_path = img_entry
        label_file = osp.join(seg_root, label_path)
        label_img = Image.open(label_file)
        edge_map = np.zeros([label_img.height, label_img.width])
        seg_map = id2trainId(np.array(label_img))

        _edgemap = edge_utils.mask_to_onehot(seg_map, num_classes)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
        edge_map = np.logical_or(edge_map, np.squeeze(_edgemap))
        edge_map_255 = np.logical_or(edge_map, np.squeeze(_edgemap))*255
        gt_img = Image.fromarray(edge_map.astype(np.uint8),mode='L')
        gt_img_255 = Image.fromarray(edge_map_255.astype(np.uint8),mode='L')
        if not osp.exists(osp.dirname(edge_root+label_path.replace('labelIds','edge_label').replace(dataset,'edge_'+dataset))):
            try:
                os.makedirs(osp.dirname(edge_root+label_path.replace('labelIds','edge_label').replace(dataset,'edge_'+dataset)))
            except Exception as e:
                print('FileExistsError')
        gt_img.save(edge_root+label_path.replace('labelIds','edge_label').replace(dataset,'edge_'+dataset))
        gt_img_255.save(edge_root+label_path.replace('labelIds','edge_label_show').replace(dataset,'edge_'+dataset))

if __name__ == '__main__':
    sc_root = '../../cityscapes/'
    edge_num_classes = 2
    num_classes = 19
    for dataset in ['train', 'val']:
        seg_root = sc_root
        edge_root = sc_root
        img_ids = [i_id.strip().split() for i_id in open(sc_root+'list/cityscapes/{}.lst'.format(dataset))]
        threads = []
        # single_core_build_edge(0, 1, dataset, seg_root, edge_root, num_classes, img_ids)
        for i in range(cpu_count()):
            threads.append(threading.Thread(target=single_core_build_edge,
                                            args=(i, cpu_count(), dataset, seg_root, edge_root, num_classes, img_ids)))
        for t in threads:
            t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()
    print('finish')
