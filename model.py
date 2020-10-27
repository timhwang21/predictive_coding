# -*- coding: utf-8 -*-
import numpy as np
import os


class Model:
    def __init__(self, dataset):
        self.dtype = np.float128

        self.dataset = dataset
        self.iteration = 30
        # self.prior = "kurtotic" # "kurtotic" or "gaussian"
        
        # NOTE: k_r (k1) and k_U (k2) do not match with that from the original paper
        self.k_r = 0.0005 # Learning rate for r
        self.k_U_init = 0.005  # Initial learning rate for U
        self.k_U = self.k_U_init

        self.sigma_sq0 = 1.0  # Variance of observation distribution of I
        self.sigma_sq1 = 10.0 # Variance of observation distribution of r1
        self.sigma_sq2 = 10.0 # Variance of observation distribution of r2
        self.sigma_sq3 = 2.0 # Variance of observation distribution of r3
        self.alpha1 = 1.0  # Precision param of r1 prior
        self.alpha2 = 0.05 # Precision param of r2 prior
        self.alpha3 = 0.05 # Precision param of r3 prior
        
        # NOTE: the original paper only provides one lambda value (which is lambda1 here)
        self.lambda1 = 0.02 # Precision param of U1 prior
        self.lambda2 = 0.00001 # Precision param of U2 prior
        self.lambda3 = 0.00001 # Precision param of U3 prior
        
        # Level 1 consists of multiple modules with 32 neurons in each moodule
        # Level-1 modules' receptive fields can be arranged into a 1D or 2D grid, with overlap between neighboring receptive fields
        self.level1_x = dataset.rf1_size[1]
        self.level1_y = dataset.rf1_size[0]
        self.level1_offset_x = dataset.rf1_offset_x
        self.level1_offset_y = dataset.rf1_offset_y
        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_size = 32

        # Level 2 consists of one module with 128 neurons whose receptive field covers that of all level-1 modules
        self.level2_module_size = 128

        # Level 3 consists of localist nodes, one for each training image for classification
        self.level3_module_size = len(dataset.images)

        # Generative Weight Matrices
        self.U1 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_y, self.level1_x, self.level1_module_size) - 0.5
        self.U2 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level2_module_size) - 0.5
        self.U3 = np.random.rand(self.level2_module_size, self.level3_module_size) - 0.5

        # State Transmission Matrices
        self.V1 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level1_module_size) - 0.5
        self.V2 = np.random.rand(self.level2_module_size, self.level2_module_size) - 0.5
        self.V3 = np.random.rand(self.level3_module_size, self.level3_module_size) - 0.5

    def prior_trans(self, x, prior):
        if prior == "kurtotic":
            x_trans = 1 + np.square(x)
        else:
            x_trans = 1
        return x_trans

    def kalman_dW(self, r0, r1, W):
        w = W.flatten()
        w_bar = np.average(w)
        w_err = w - w_bar
        w_cov = w_err @ w_err.T
        
        r0 = r0.flatten()
        R1 = np.zeros([r0.size, w.size])
        for i in np.arange(r0.size):
            R1[i, r1.size*i:r1.size*(i+1)] = r1[:,None].T
        
        r10 = R1 @ w
        e10 = r0 - r10
        cov10 = e10.flatten() @ e10.flatten().T
        
        N_inv = R1.T.dot(R1)/cov10
        N = np.linalg.inv(N_inv + 1/w_cov * np.eye(N_inv.shape[0])*10**-10)
        
        dW = (N.dot(R1.T)/cov10 @ e10).reshape(W.shape)
        
        return dW

    def kalman_dr(self, r0, r1, r2, U1, V1, U2, categorical=False):
        r0_x = r0.flatten()
        r1_x = r1.flatten()
        r2_x = r2.flatten()
        
        U1_x = U1.reshape(r0_x.size, r1_x.size)
        V1_x = V1.reshape(r1_x.size, r1_x.size)
        U2_x = U2.reshape(r1_x.size, r2_x.size)
        
        r10 = U1_x @ r1_x
        r11 = V1_x @ r1_x
        r21 = U2_x @ r2_x
        
        e10 = r0_x - r10
        e11 = r1_x - r11
        e21 = r1_x - r21 if not categorical else (np.exp(1)/np.sum(np.exp(r1_x))) - r21
        
        cov10 = e10 @ e10.T
        cov11 = e11 @ e11.T
        cov21 = e21 @ e21.T
        
        N_inv = U1_x.T.dot(U1_x)/cov10 + 1/cov11 + 1/cov21
        N = np.linalg.inv(N_inv + np.eye(N_inv.shape[0])*10**-10)
        r_x =  N @ (U1_x.T.dot(r0_x)/cov10 + r11/cov11 + r21/cov21)

        dr = r_x.reshape(r1.shape) - r11

        return dr

    # inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
    # labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
    def apply_input(self, inputs, labels, dataset, training):
        outputs = {'index':[],
                   'input': [],
                   'label': [],
                   'iteration': [],
                   'r1': [],
                   'r2': [],
                   'r3': []}

        # inputs and estimates
        inputs = np.array(inputs, dtype=self.dtype)
        r1 = np.random.normal(loc=0.0, scale=0.01, size=(self.level1_layout_y, self.level1_layout_x, self.level1_module_size)).astype(self.dtype)
        r2 = np.random.normal(loc=0.0, scale=0.01, size=self.level2_module_size).astype(self.dtype)
        r3 = np.random.normal(loc=0.0, scale=0.01, size=self.level3_module_size).astype(self.dtype)
    
        for idx in np.ndindex(inputs.shape[:2]):
            inputs_idx = dataset.get_rf1_patches_from_rf2_patch(inputs[idx])
            labels_idx = labels[idx] if training else labels[idx]*0

            for i in range(self.iteration):
                # calculate r updates
                dr1 = np.array([self.kalman_dr(inputs_idx[j,k], r1[j,k], r2, self.U1[j,k], self.V1[j,k], self.U2[j,k]) for j,k in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(r1.shape)
                dr2 = self.kalman_dr(r1, r2, r3, self.U2, self.V2, self.U3)
                dr3 = self.kalman_dr(r2, r3, labels_idx, self.U3, self.V3, np.eye(len(labels_idx)), categorical=True)

                # calculate U and V updates
                if training:
                    dU1 = np.array([self.kalman_dW(inputs_idx[j,k], r1[j,k], self.U1[j,k]) for j,k in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(self.U1.shape)
                    dU2 = np.array([self.kalman_dW(r1[j,k], r2, self.U2[j,k]) for j,k in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(self.U2.shape)
                    dU3 = self.kalman_dW(r2, r3, self.U3)
                    dV1 = np.array([self.kalman_dW(r1[j,k], r1[j,k], self.V1[j,k]) for j,k in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(self.V1.shape)
                    dV2 = self.kalman_dW(r2, r2, self.V2)
                    dV3 = self.kalman_dW(r3, r3, self.V3)

                # apply r updates
                r1 += dr1
                r2 += dr2
                r3 += dr3

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
