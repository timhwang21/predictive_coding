# -*- coding: utf-8 -*-
import numpy as np
import os


class Model:
    def __init__(self, dataset, iteration=30, prior="kurtotic"):
        self.dataset = dataset
        self.iteration = iteration
        self.prior = prior # "kurtotic" or "gaussian"
        
        # NOTE: k_r (k1) and k_U (k2) do not match with that from the original paper
        self.k_r = 0.0005 # Learning rate for r
        self.k_U_init = 0.005  # Initial learning rate for U
        # k_U (k2) decreases gradually by dividing with 1.015 every 40 training inputs
        self.k_U_decay_cycle = 40
        self.k_U_decay_rate = 1.015

        self.sigma_sq0 = 1.0  # Variance of observation distribution of I
        self.sigma_sq1 = 10.0 # Variance of observation distribution of r1
        self.sigma_sq2 = 10.0 # Variance of observation distribution of r2
        self.sigma_sq3 = 2.0 # Variance of observation distribution of r3
        self.alpha1 = 1.0  # Precision param of r1 prior
        self.alpha2 = 0.05 # Precision param of r2 prior
        self.alpha3 = 0.05 # Precision param of r3 prior
        
        # NOTE: the original paper only provides one lambda value (which is lambda1 here)
        self.lambda1 = 0.02 # Precision param of U1 prior    (var=50.0, std=7.1)
        self.lambda2 = 0.00001 # Precision param of U2 prior
        self.lambda3 = 0.00001
        
        # NOTE: NOT SURE WHAT THIS SCALE IS FOR
        # U: weight matrix
        U_scale = 1.0
        
        # inputs: three 16 x 16 overlapping rf1 patches (offset by 5 pixels horizontally)
        # Level 1: 3 modules, each module has 32 input-estimating neurons and 32 error-detecting neurons
        # Level 2: 128 input-estimating neurons and 128 error-detecting neurons
        self.input_x = dataset.rf1_size[1]
        self.input_y = dataset.rf1_size[0]
        self.input_size = self.input_x * self.input_y
        self.input_offset_x = dataset.rf1_offset_x
        self.input_offset_y = dataset.rf1_offset_y

        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_n = self.level1_layout_x * self.level1_layout_y
        self.level1_module_size = 32
        
        self.level2_module_size = 128

        # U1: level-1 top-down weights
        # 3 modules (one for each rf1 patch); 256 pixels (16 x 16) in each rf1 patch; 32 input-estimating neurons in each module
        self.U1 = (np.random.rand(self.level1_module_n, self.input_size, self.level1_module_size) - 0.5) * U_scale
        # U2: level-2 top-down weights
        # 96 (3 x 32) level-1 error-detecting neurons; 128 level-2 input-estimating neurons
        self.U2 = (np.random.rand(self.level1_module_n * self.level1_module_size, self.level2_module_size) - 0.5) * U_scale

        self.k_U = self.k_U_init

        # level 3 for classification (level-3 consists of localist nodes, one for each training image)
        self.level3_module_size = len(dataset.images)
        self.U3 = (np.random.rand(self.level2_module_size, self.level3_module_size) - 0.5) * U_scale

    def prior_trans(self, x, prior):
        if prior == "kurtotic":
            x_trans = 1 + np.square(x)
        else:
            x_trans = 1
        return x_trans

    def apply_input(self, inputs, label, training):
        inputs = np.array(inputs)
        r1 = np.zeros((self.level1_module_n, self.level1_module_size), dtype=np.float32)
        r2 = np.zeros(self.level2_module_size, dtype=np.float32)
        r3 = np.zeros(self.level3_module_size, dtype=np.float32)
    
        for i in range(self.iteration):
            # predictions
            r10 = np.matmul(self.U1, r1[:, :, None]).squeeze()
            r21 = self.U2.dot(r2).reshape(r1.shape)
            r32 = self.U3.dot(r3)
            r43 = label if training else label*0

            # prediction errors
            e0 = inputs - r10
            e1 = r1 - r21
            e2 = r2 - r32
            e3 = (np.exp(r3)/np.sum(np.exp(r3))) - r43 # softmax cross-entropy loss

            # r updates
            dr1 = (self.k_r/self.sigma_sq0) * np.matmul(np.transpose(self.U1, axes=(0,2,1)), e0[:, :, None]).squeeze() \
                  + (self.k_r/self.sigma_sq1) * -e1 \
                  - self.k_r * self.alpha1 * r1 / self.prior_trans(r1, self.prior)

            dr2 = (self.k_r / self.sigma_sq1) * self.U2.T.dot(e1.flatten()) \
                  + (self.k_r / self.sigma_sq2) * -e2 \
                  - self.k_r * self.alpha2 * r2 / self.prior_trans(r2, self.prior)

            dr3 = (self.k_r / self.sigma_sq2) * self.U3.T.dot(e2) \
                + (self.k_r / self.sigma_sq3) * -e3 \
                  - self.k_r * self.alpha3 * r3 / self.prior_trans(r3, self.prior)

            # U updates
            if training:
                dU1 = (self.k_U/self.sigma_sq0) * np.matmul(e0[:, :, None], r1[:, None, :]) \
                       - self.k_U * self.lambda1 * self.U1 / self.prior_trans(self.U1, self.prior)

                dU2 = (self.k_U / self.sigma_sq1) * np.outer(e1.flatten(), r2) \
                      - self.k_U * self.lambda2 * self.U2 / self.prior_trans(self.U2, self.prior)

                dU3 = (self.k_U / self.sigma_sq2) * np.outer(e2, r3) \
                      - self.k_U * self.lambda3 * self.U3 / self.prior_trans(self.U3, self.prior)

            # apply r updates
            r1 += dr1
            r2 += dr2
            r3 += dr3

            # apply U updates
            if training:
                self.U1 += dU1
                self.U2 += dU2
                self.U3 += dU3

        # flatten level 1 nodes to vectors
        r1 = r1.flatten()
        e1 = e1.flatten()

        return r1, r2, r3, e1, e2, e3

    def train(self, dataset):
        rf2_patch_n = len(dataset.rf2_patches) # 2375

        for i in range(rf2_patch_n):
            # Loop for all rf2 patches
            rf1_patches = dataset.get_rf1_patches(i)
            label = dataset.labels[i]
            r1, r2, r3, e1, e2, e3 = self.apply_input(rf1_patches, label, training=True)

        print("train finished")

    # representations at each level
    # values change with input
    def reconstruct(self, r, level=1):
        if level==1:
            r1 = r # (96,)
        elif level==2:
            r2 = r # (128,)
            r1 = self.U2.dot(r2) # (96,)
        elif level==3:
            r3 = r
            r2 = self.U3.dot(r3)
            r1 = self.U2.dot(r2)
            
        # reconstructed image size is 16 x 26 because the each set of inputs is three overlapping (offset by 5 pixels horizontally) 16 x 16 rf1 patches
        rf2_patch = np.zeros((self.input_y + (self.input_offset_y * (self.level1_layout_y - 1)), \
                              self.input_x + (self.input_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)
        
        # reconstruct each of the three rf1 patches separately and then combine
        for i in range(self.level1_module_n):
            module_y = i % self.level1_layout_y
            module_x = i // self.level1_layout_y

            r = r1[self.level1_module_size * i:self.level1_module_size * (i+1)]
            U = self.U1[i]
            Ur = U.dot(r).reshape(self.input_y, self.input_x)
            rf2_patch[self.input_offset_y * module_y :self.input_offset_y * module_y + self.input_y, \
                      self.input_offset_x * module_x :self.input_offset_x * module_x + self.input_x] += Ur
        return rf2_patch

    # rf: receptive field
    # values don't change with input (when model is not being trained)
    def get_level2_rf(self, index):
        rf = np.zeros((self.input_y + (self.input_offset_y * (self.level1_layout_y - 1)), \
                       self.input_x + (self.input_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)

        for i in range(self.level1_module_n):
            module_y = i % self.level1_layout_y
            module_x = i // self.level1_layout_y

            U2 = self.U2[:,index][self.level1_module_size * i:self.level1_module_size * (i+1)]
            UU = self.U1[i].dot(U2).reshape((self.input_y, self.input_x))
            rf[self.input_offset_y * module_y :self.input_offset_y * module_y + self.input_y, \
               self.input_offset_x * module_x :self.input_offset_x * module_x + self.input_x] += UU

        return rf

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            U1=self.U1,
                            U2=self.U2,
                            U3=self.U3)
        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        data = np.load(file_path)
        self.U1 = data["U1"]
        self.U2 = data["U2"]
        self.U3 = data["U3"]
        print("loaded: {}".format(dir_name))
