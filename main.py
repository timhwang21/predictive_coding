# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import imageio

from dataset import Dataset
from model import Model

def main():
    # dataset = Dataset(scale=1.0, shuffle=False)
    dataset = Dataset(scale=1.0, shuffle=False, data_dir="./data/images_rao_128x128",
                      rf1_x=16, rf1_y=16, rf1_offset_x=8, rf1_offset_y=8,
                      rf1_layout_x=15, rf1_layout_y=15, gauss_mask_sigma=1.0)
    model = Model(iteration=500, dataset=dataset)
    model.train(dataset)

    model.save("saved")

    if not os.path.exists("result"):
        os.mkdir("result")
    
    # receptive fields of neurons in level 1 center module
    for i in range(model.level1_module_size):
        u1 = model.Us[model.level1_module_n//2][:,i].reshape((dataset.rf1_size[1],dataset.rf1_size[0]))
        u1 = cv2.resize(u1, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("result/u1_{:0>2}.png".format(i), u1)
    
    # receptive fields of neurons in level 2
    for i in range(model.level2_module_size):
        u2 = model.get_level2_rf(i)
        u2 = cv2.resize(u2, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("result/u2_{:0>3}.png".format(i), u2)
    
    # difference-of-gaussian filtered image inputs
    for i in range(len(dataset.filtered_images)):
        filtered_img = dataset.filtered_images[i]
        filtered_img = cv2.resize(filtered_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("result/input_image{:0>2}.png".format(i), filtered_img)

    # image reconstruction based on level 1 and level 2 representations
    for i in range(len(dataset.images)):
        images = dataset.get_images(i)
        label = dataset.labels[i]
        rs, r_tds, rh, error_tds, r3 = model.apply_images(images, label, training=False)
        print("Target vector:", label)
        print("Level 3 activation vector:", r3)
        print("Most active node in level 3 vector:", np.argmax(r3))
        level1_img = model.reconstruct(rs, level=1)
        level2_img = model.reconstruct(rh, level=2)
        level1_img = cv2.resize(level1_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        level2_img = cv2.resize(level2_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("result/level1_image{:0>2}.png".format(i), level1_img)
        imageio.imwrite("result/level2_image{:0>2}.png".format(i), level2_img)

if __name__ == '__main__':
    main()
