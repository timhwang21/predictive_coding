# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy.special import expit

class Model:
    def __init__(self, dataset, level1_module_size=32, level2_module_size=128):
        self.dtype = np.float32

        self.dataset = dataset
        self.iteration = 30
        
        # Level 1 consists of multiple modules with 32 neurons in each moodule
        # Level-1 modules' receptive fields can be arranged into a 1D or 2D grid, with overlap between neighboring receptive fields
        self.level1_x = dataset.rf1_size[1]
        self.level1_y = dataset.rf1_size[0]
        self.level1_offset_x = dataset.rf1_offset_x
        self.level1_offset_y = dataset.rf1_offset_y
        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_size = level1_module_size

        # Level 2 consists of one module with 128 neurons whose receptive field covers that of all level-1 modules
        self.level2_module_size = level2_module_size

        # Level 3 consists of localist nodes, one for each training image for classification
        self.level3_module_size = len(dataset.images)

        # Generative Weight Matrices
        self.U1 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_y, self.level1_x, self.level1_module_size).astype(self.dtype) - 0.5
        self.U2 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level2_module_size).astype(self.dtype) - 0.5
        self.U3 = np.random.rand(self.level2_module_size, self.level3_module_size).astype(self.dtype) - 0.5

        # State Transmission Matrices
        self.V1 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level1_module_size).astype(self.dtype) - 0.5
        self.V2 = np.random.rand(self.level2_module_size, self.level2_module_size).astype(self.dtype) - 0.5
        self.V3 = np.random.rand(self.level3_module_size, self.level3_module_size).astype(self.dtype) - 0.5

        # Learning Rates
        self.alpha_r = 1e-3
        self.alpha_u = 1e-3
        self.alpha_v = 1e-3

    def kalman_dW(self, r0, r1, e10, alpha):
        e10 = (1 - np.square(r0)) * e10
        dW = alpha * np.outer(e10, r1)
        return dW

    def kalman_dr(self, U1, r0, r11, r21, alpha):
        e10_bar = r0 - U1 @ r11
        e21_bar = r11 - r21
        e10_bar = (1 - np.square(r0)) * e10_bar
        e21_bar = (1 - np.square(r11)) * e21_bar
        dr1 = alpha * (U1.T @ e10_bar) - alpha * e21_bar
        return dr1

    # inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
    # labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
    def apply_input(self, inputs, labels, dataset, training):
        outputs = {'index':[],
                   'input': [],
                   'label': [],
                   'iteration': [],
                   'r1': [],
                   'r2': [],
                   'r3': [],
                   'e10': [],
                   'e21': [],
                   'e32': [],
                   'e43': [],
                   'e11': [],
                   'e22': [],
                   'e33': []}

        # inputs and estimates
        inputs = np.array(inputs, dtype=self.dtype)
        r1 = np.random.normal(loc=0.0, scale=0.01, size=(self.level1_layout_y, self.level1_layout_x, self.level1_module_size)).astype(self.dtype)
        r2 = np.random.normal(loc=0.0, scale=0.01, size=self.level2_module_size).astype(self.dtype)
        r3 = np.random.normal(loc=0.0, scale=0.01, size=self.level3_module_size).astype(self.dtype)
    
        # before receiving input
        ## within-level predictions for next time step
        r11 = np.matmul(self.V1, np.expand_dims(r1, axis=-1)).reshape(r1.shape)
        r22 = self.V2 @ r2
        r33 = self.V3 @ r3

        for idx in np.ndindex(inputs.shape[:2]):
            I = dataset.get_rf1_patches_from_rf2_patch(inputs[idx])
            L = labels[idx]

            for i in range(self.iteration):
                # reshape
                I_x = I.reshape(I.shape[:2] + (-1,))
                U1_x = self.U1.reshape(I_x.shape + (r1.shape[-1],))

                # predictions
                ## between-level
                r10 = np.matmul(U1_x, np.expand_dims(r1, axis=-1)).reshape(I_x.shape)
                r21 = np.matmul(self.U2, np.expand_dims(r2, axis=-1)).reshape(r1.shape)
                r32 = self.U3 @ r3
                r43 = L if training else L*0

                # tanh transformation
                r10 = np.tanh(r10)
                r21 = np.tanh(r21)
                r32 = np.tanh(r32)
                r11 = np.tanh(r11)
                r22 = np.tanh(r22)
                r33 = np.tanh(r33)

                # prediction errors
                ## between-level
                e10 = I_x - r10
                e21 = r1 - r21
                e32 = r2 - r32
                e43 = (expit(r3)/np.sum(expit(r3))) - r43
                ## within-level
                e11 = r1 - r11
                e22 = r2 - r22
                e33 = r3 - r33

                # calculate r updates
                dr1 = np.array([self.kalman_dr(U1_x[j,k], I_x[j,k], r11[j,k], r21[j,k], alpha = self.alpha_r) for j,k in np.ndindex(I.shape[:2])]).reshape(r1.shape)
                dr2 = sum([self.kalman_dr(self.U2[j,k], r1[j,k], r22, r32, alpha = self.alpha_r) for j,k in np.ndindex(I.shape[:2])])
                dr3 = self.kalman_dr(self.U3, r2, r33, r43, alpha = self.alpha_r)

                # calculate U and V updates
                if training:
                    dU1 = np.array([self.kalman_dW(I_x[j,k], r1[j,k], e10[j,k], alpha = self.alpha_u) for j,k in np.ndindex(I.shape[:2])]).reshape(self.U1.shape)
                    dU2 = np.array([self.kalman_dW(r1[j,k], r2, e21[j,k], alpha = self.alpha_u) for j,k in np.ndindex(I.shape[:2])]).reshape(self.U2.shape)
                    dU3 = self.kalman_dW(r2, r3, e32, alpha = self.alpha_u)

                    dV1 = np.array([self.kalman_dW(r1[j,k], r1[j,k], e11[j,k], alpha = self.alpha_v) for j,k in np.ndindex(I.shape[:2])]).reshape(self.V1.shape)
                    dV2 = self.kalman_dW(r2, r2, e22, alpha = self.alpha_v)
                    dV3 = self.kalman_dW(r3, r3, e33, alpha = self.alpha_v)

                # apply r updates
                r1 = r11 + dr1
                r2 = r22 + dr2
                r3 = r33 + dr3

                # within-level predictions for next time step
                r11 = np.matmul(self.V1, np.expand_dims(r1, axis=-1)).reshape(r1.shape)
                r22 = self.V2 @ r2
                r33 = self.V3 @ r3

                # apply U and V updates
                if training:
                    self.U1 += dU1
                    self.U2 += dU2
                    self.U3 += dU3
                    self.V1 += dV1
                    self.V2 += dV2
                    self.V3 += dV3
                
                # append outputs
                outputs['index'].append(idx)
                outputs['input'].append(inputs[idx])
                outputs['label'].append(labels[idx])
                outputs['iteration'].append(i)
                outputs['r1'].append(r1.copy())
                outputs['r2'].append(r2.copy())
                outputs['r3'].append(r3.copy())
                outputs['e10'].append(e10.copy())
                outputs['e21'].append(e21.copy())
                outputs['e32'].append(e32.copy())
                outputs['e43'].append(e43.copy())
                outputs['e11'].append(e11.copy())
                outputs['e22'].append(e22.copy())
                outputs['e33'].append(e33.copy())

        return outputs

    # training on rf2_patches in order within a given image
    def train(self, dataset):
        # images are presented in the order defined in dataset
        # rf2 patches of a given image are presented in an ascending sequence (moves through x before y)
        # inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
        # labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
        for i in range(dataset.rf2_patches.shape[0]):
            inputs = dataset.rf2_patches[i]
            labels = dataset.labels[i]
            outputs = self.apply_input(inputs, labels, dataset, training=True)

        print("train finished")

    # representations at each level
    # values change with input
    def reconstruct(self, r, level=1):
        if level == 1:
            r1 = r
        elif level == 2:
            r2 = r
            r1 = self.U2.dot(r2).reshape((self.level1_layout_y, self.level1_layout_x, self.level1_module_size))
        elif level == 3:
            r3 = r
            r2 = self.U3.dot(r3)
            r1 = self.U2.dot(r2).reshape((self.level1_layout_y, self.level1_layout_x, self.level1_module_size))
            
        rf2_patch = np.zeros((self.level1_y + (self.level1_offset_y * (self.level1_layout_y - 1)), \
                              self.level1_x + (self.level1_offset_x * (self.level1_layout_x - 1))), dtype=self.dtype)
        
        # reconstruct each of the three rf1 patches separately and then combine
        for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x):
            r = r1[i, j]
            U = self.U1[i, j]
            Ur = U.dot(r)
            rf2_patch[self.level1_offset_y * i : self.level1_offset_y * i + self.level1_y, \
                      self.level1_offset_x * j : self.level1_offset_x * j + self.level1_x] += Ur
        return rf2_patch

    # rf: receptive field
    # values don't change with input (when model is not being trained)
    def get_level2_rf(self, index):
        rf = np.zeros((self.level1_y + (self.level1_offset_y * (self.level1_layout_y - 1)), \
                       self.level1_x + (self.level1_offset_x * (self.level1_layout_x - 1))), dtype=self.dtype)

        for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x):
            U2 = self.U2[i, j, :, index]
            UU = self.U1[i, j].dot(U2)
            rf[self.level1_offset_y * i :self.level1_offset_y * i + self.level1_y, \
               self.level1_offset_x * j :self.level1_offset_x * j + self.level1_x] += UU

        return rf

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            U1=self.U1,
                            U2=self.U2,
                            U3=self.U3,
                            V1=self.V1,
                            V2=self.V2,
                            V3=self.V3)
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
        self.V1 = data["V1"]
        self.V2 = data["V2"]
        self.V3 = data["V3"]
        print("loaded: {}".format(dir_name))
