# -*- coding: utf-8 -*-
#
# Copyright 2018 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
kraken.lib.zca
~~~~~~~~~~~~~~~~~

Some routines for ZCA whitening of images.
"""
import numpy as np
from numpy import linalg

def zca_fit(im, eps=10e-7):
    """
    Calculates the principal components of image array of shape (N, C, H, W)
    returning a whitening matrix.
    """
    # flatten 
    fim = im.reshape(im.shape[0], -1)
    # estimate covariance
    cov = np.dot(fim.T, fim)/fim.shape[0]
    u, s, _ = linalg.svd(cov)
    s = 1. / np.sqrt(s[np.newaxis] + eps)
    return (u*s).dot(u.T)
