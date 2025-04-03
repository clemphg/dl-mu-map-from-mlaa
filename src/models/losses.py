
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

import numpy as np
import math


def central_finite_diff_first_derivative(accuracy):
    """Tabulated first order derivatives of the central finite difference.
    https://github.com/j-onofrey/deep-image-pet/blob/main/estimator/util/LayerUtil.py#L449
    """
    if accuracy not in [2, 4, 6, 8]:
        raise ValueError('Accuracy must be 2, 4, 6 or 8')

    dev =   {
            2 : ([-1/2, 0.0, 1/2], 1),
            4 : ([1/12, -2/3, 0.0, 2/3, -1/12], 2),
            6 : ([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60], 3),
            8 : ([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280], 5) 
        }

    return dev[accuracy]


def image_gradient(input_data, accuracy=4, visualize=False):
    """Computes the image gradient for 2D or 3D images."""
    num_channels = input_data.shape[1]  # Assume channels first format (N, C, (D,) H, W)

    filt, filt_radius = central_finite_diff_first_derivative(accuracy) # returns ([1/12, -2/3, 0.0, 2/3, -1/12], 2)
    length = len(filt)

    filt = torch.tensor(filt, dtype=torch.float32, device=input_data.device).repeat(num_channels, 1, 1)

    real_dims = len(input_data.shape) - 2

    if real_dims == 2:
        # 2D convolution kernels
        kernel_x = filt.view(1, num_channels, length, 1).repeat(num_channels, 1, 1, 1)
        kernel_y = filt.view(1, num_channels, 1, length).repeat(num_channels, 1, 1, 1)

        # Padding for convolution
        pad_x = (filt_radius, filt_radius, 0, 0)  # (left, right, top, bottom)
        pad_y = (0, 0, filt_radius, filt_radius)

        # Apply padding
        input_data_x = nn.functional.pad(input_data, pad_x, mode='reflect')
        input_data_y = nn.functional.pad(input_data, pad_y, mode='reflect')

        # Compute gradients
        gradient_x = nn.functional.conv2d(input_data_x, kernel_x, groups=num_channels)
        gradient_y = nn.functional.conv2d(input_data_y, kernel_y, groups=num_channels)

        return {'x': gradient_x, 'y': gradient_y}

    elif real_dims == 3:
        # 3D convolution kernels
        kernel_x = filt.view(1, num_channels, length, 1, 1).repeat(num_channels, 1, 1, 1, 1)
        kernel_y = filt.view(1, num_channels, 1, length, 1).repeat(num_channels, 1, 1, 1, 1)
        kernel_z = filt.view(1, num_channels, 1, 1, length).repeat(num_channels, 1, 1, 1, 1)

        # Padding
        pad_x = (0, 0, 0, 0, filt_radius, filt_radius)
        pad_y = (0, 0, filt_radius, filt_radius, 0, 0)
        pad_z = (filt_radius, filt_radius, 0, 0, 0, 0)

        # Apply padding
        input_data_x = nn.functional.pad(input_data, pad_x, mode='reflect')
        input_data_y = nn.functional.pad(input_data, pad_y, mode='reflect')
        input_data_z = nn.functional.pad(input_data, pad_z, mode='reflect')

        # Compute gradients
        gradient_x = nn.functional.conv3d(input_data_x, kernel_x, groups=num_channels, stride=1, padding=0)
        gradient_y = nn.functional.conv3d(input_data_y, kernel_y, groups=num_channels, stride=1, padding=0)
        gradient_z = nn.functional.conv3d(input_data_z, kernel_z, groups=num_channels, stride=1, padding=0)
        
        return {'x': gradient_x, 'y': gradient_y, 'z': gradient_z}
    else:
        raise ValueError(f"Cannot calculate image gradient for {real_dims}-dimensional data")

def image_gradient_difference_loss(output_layer, labels, exponent=2):
    """Computes the image gradient difference loss."""
    if exponent < 1:
        raise ValueError("Exponent cannot be less than 1 for GDL")
    
    n_dims = len(output_layer.shape) - 2
    gradients_recon = image_gradient(output_layer)
    gradients_target = image_gradient(labels)
    
    if n_dims == 2:
        grad_diff_x = torch.abs(gradients_target['x'] - gradients_recon['x'])
        grad_diff_y = torch.abs(gradients_target['y'] - gradients_recon['y'])
        return torch.mean(grad_diff_x ** exponent + grad_diff_y ** exponent)
    
    elif n_dims == 3:
        grad_diff_x = torch.abs(gradients_target['x'] - gradients_recon['x'])
        grad_diff_y = torch.abs(gradients_target['y'] - gradients_recon['y'])
        grad_diff_z = torch.abs(gradients_target['z'] - gradients_recon['z'])
        return torch.mean(grad_diff_x ** exponent + grad_diff_y ** exponent + grad_diff_z ** exponent)
    
    else:
        raise ValueError("GDL is only supported for 2D or 3D data")


def rotate_3d_volume(volume, angle):
    """Rotates each depth slice (D) in a 3D volume individually."""
    N, C, D, H, W = volume.shape
    rotated_slices = [F.rotate(volume[:, :, i, :, :], angle, interpolation=T.InterpolationMode.BILINEAR) for i in range(D)]
    return torch.stack(rotated_slices, dim=2)


def line_integral_loss(output_layer, labels, num_angles=4):
    """LIP-loss"""
    angle_inc = 180.0 / num_angles
    LI_loss = 0.0
    
    for rot_ind in range(num_angles):
        rot_angle = math.radians(rot_ind * angle_inc)
        output_rotated = rotate_3d_volume(output_layer, rot_angle)
        labels_rotated = rotate_3d_volume(labels, rot_angle)
        
        output_proj = output_rotated.sum(dim=1)
        label_proj = labels_rotated.sum(dim=1)
        
        error_proj = torch.mean((output_proj - label_proj) ** 2)
        LI_loss += error_proj
    
    return LI_loss / num_angles