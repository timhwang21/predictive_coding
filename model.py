# -*- coding: utf-8 -*-
import numpy as np
import os


class Model:
    def __init__(self, dataset, iteration=30):
        self.dataset = dataset
        self.iteration = iteration
        
        # NOTE: k_r (k1) and k_U (k2) do not match with that from the original paper
        self.k_r = 0.0005 # Learning rate for r
        self.k_U_init = 0.005  # Initial learning rate for U
        # k_U (k2) decreases gradually by dividing with 1.015 every 40 training inputs
        self.k_U_decay_cycle = 40
        self.k_U_decay_rate = 1.015

        self.sigma_sq0 = 1.0  # Variance of observation distribution of I
        self.sigma_sq1 = 10.0 # Variance of observation distribution of r
        self.sigma_sq2 = 10.0
        self.sigma_sq3 = 10.0
        self.alpha1 = 1.0  # Precision param of r prior    (var=1.0,  std=1.0)
        self.alpha2 = 0.05 # Precision param of r_td prior (var=20.0, std=4.5)
        self.alpha3 = 0.05
        
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

        # NOTE: NOT SURE WHAT THIS SCALE IS FOR
        # Scaling parameter for learning rate of level2
        self.level2_lr_scale = 1.0

        # level 3 for classification (level-3 consists of localist nodes, one for each training image)
        self.level3_module_size = len(dataset.images)
        self.U3 = (np.random.rand(self.level2_module_size, self.level3_module_size) - 0.5) * U_scale

    def apply_input(self, inputs, label, training):
        # 96 (3 x 32) level-1 input-estimating neurons' representations
        r1 = np.zeros([self.level1_module_n * self.level1_module_size], dtype=np.float32)
        # 128 level-2 input-estimating neurons' representations
        r2 = np.zeros([self.level2_module_size], dtype=np.float32)
        # level-3 representations
        r3 = np.zeros([self.level3_module_size], dtype=np.float32)
        
        # e1 (r1-r21): 96 (3 x 32) level 2's bottom-up prediction error (from level 1 to level 2)
        e1 = np.zeros([self.level1_module_n * self.level1_module_size], dtype=np.float32)
    
        for i in range(self.iteration):
            # Loop for iterations

            # Calculate r21 (level 2's prediction for level 1)
            r21 = self.U2.dot(r2) # (96,)

            for m in range(self.level1_module_n):
                # corresponding neurons in a given level-1 module
                m_start = self.level1_module_size * m
                m_end = self.level1_module_size * (m+1)
                # inputs
                I = inputs[m]
                # level-1 input estimates
                r1_m = r1[m_start:m_end]
                # level 2's predictions for level 1
                r21_m = r21[m_start:m_end]

                # weights from level 1 to level 0 (input)
                U1_m = self.U1[m] # (256,32)
                # level 1's predictions for input
                r10_m = U1_m.dot(r1_m) # (256,)

                # level 1's bottom-up prediction error (from input to level 1)
                e0_m = I - r10_m # (256,)
                # e1_m: level 2's bottom-up prediction error (from level 1 to level 2)
                # -e1_m: level 1's within-level prediction error (based on level 2's predictions)
                e1_m = r1_m - r21_m # (32,)
                
                # gradient descent on E (optimization function) with respect to r, assuming Gaussian prior distribution
                # Equation 7
                # NOTE: seems to assume the activation function is linear here (f(x) = x) instead of tanh(x)
                dr1_m = (self.k_r/self.sigma_sq0) * U1_m.T.dot(e0_m) \
                     + (self.k_r/self.sigma_sq1) * -e1_m \
                     - self.k_r * self.alpha1 * r1_m
                     # (32,)

                # gradient descent on E (optimization function) with respect to U
                # Equation 9
                if training:
                    dU1_m = (self.k_U/self.sigma_sq0) * np.outer(e0_m, r1_m) \
                         - self.k_U * self.lambda1 * U1_m
                         # (256,32)
                    self.U1[m] += dU1_m

                r1[m_start:m_end] += dr1_m
                e1[m_start:m_end] = e1_m

            # Level2 update
            r32 = self.U3.dot(r3)
            e2 = r2 - r32 # (128,)
            e3 = r3 - label

            # gradient descent on E (optimization function) with respect to r, assuming Gaussian prior distribution
            # Equation 7
            dr2 = (self.k_r*self.level2_lr_scale / self.sigma_sq1) * self.U2.T.dot(e1) \
                  + (self.k_r*self.level2_lr_scale / self.sigma_sq2) * -e2 \
                  - self.k_r*self.level2_lr_scale * self.alpha2 * r2
                  # (128,)
            
            # gradient descent on E (optimization function) with respect to U
            # Equation 9
            if training:
                dU2 = (self.k_U*self.level2_lr_scale / self.sigma_sq1) * np.outer(e1, r2) \
                      - self.k_U*self.level2_lr_scale * self.lambda2 * self.U2
                # (96,128)
                self.U2 += dU2

            r2 += dr2

            # level 3 (classification) update
            dr3 = (self.k_r / self.sigma_sq2) * self.U3.T.dot(e2) \
                + (self.k_r / self.sigma_sq3) * -e3 \
                  - self.k_r * self.alpha3 * r3
            
            if training:
                dU3 = (self.k_U / self.sigma_sq2) * np.outer(e2, r3) \
                      - self.k_U * self.lambda3 * self.U3
                
                self.U3 += dU3
            
            r3 += dr3
            r3 = np.exp(r3)/sum(np.exp(r3)) # softmax

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
