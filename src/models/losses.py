
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def central_finite_diff_first_derivative(accuracy):
    """Returns the finite difference kernel and its radius."""
    if accuracy == 2:
        filt = [-0.5, 0, 0.5]
    elif accuracy == 4:
        filt = [1/12, -2/3, 0, 2/3, -1/12]
    else:
        raise ValueError("Unsupported accuracy")
    return filt, len(filt) // 2

def image_gradient(input_data, accuracy=4, visualize=False):
    """Computes the image gradient for 2D or 3D images."""
    num_channels = input_data.shape[1]  # Assume channels first format (N, C, H, W)
    filt, filt_radius = central_finite_diff_first_derivative(accuracy)
    filt = np.array(filt, dtype=np.float32)
    filt = np.tile(filt, (num_channels, 1, 1))  # Expand for channels
    filt = torch.tensor(filt, dtype=torch.float32, device=input_data.device)

    real_dims = len(input_data.shape) - 2
    
    if real_dims == 2:
        kernel_x = filt.view(1, num_channels, len(filt), 1)
        kernel_y = filt.view(1, num_channels, 1, len(filt))
        
        gradient_x = F.conv2d(input_data, kernel_x, padding=(filt_radius, 0), groups=num_channels)
        gradient_y = F.conv2d(input_data, kernel_y, padding=(0, filt_radius), groups=num_channels)
        
        return {'x': gradient_x, 'y': gradient_y}
    
    elif real_dims == 3:
        kernel_x = filt.view(1, num_channels, len(filt), 1, 1)
        kernel_y = filt.view(1, num_channels, 1, len(filt), 1)
        kernel_z = filt.view(1, num_channels, 1, 1, len(filt))
        
        gradient_x = F.conv3d(input_data, kernel_x, padding=(filt_radius, 0, 0), groups=num_channels)
        gradient_y = F.conv3d(input_data, kernel_y, padding=(0, filt_radius, 0), groups=num_channels)
        gradient_z = F.conv3d(input_data, kernel_z, padding=(0, 0, filt_radius), groups=num_channels)
        
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


def line_integral_loss(output_layer, labels, num_angle):
    """Computes the line integral loss."""
    output_layer_squeeze = output_layer.squeeze(-1)
    labels_squeeze = labels.squeeze(-1)
    
    angle_inc = 180.0 / num_angle
    LI_loss = 0.0
    
    for rot_ind in range(num_angle):
        rot_angle = math.radians(rot_ind * angle_inc)
        output_rotated = F.rotate(output_layer_squeeze, rot_angle, mode='bilinear')
        labels_rotated = F.rotate(labels_squeeze, rot_angle, mode='bilinear')
        
        output_proj = output_rotated.sum(dim=1)
        label_proj = labels_rotated.sum(dim=1)
        
        error_proj = torch.mean((output_proj - label_proj) ** 2)
        LI_loss += error_proj
    
    return LI_loss / num_angle