"""
Non-TOF and TOF PET projector.

Using ParallelProj (https://parallelproj.readthedocs.io/en/stable/).
"""

import torch
import parallelproj

import array_api_compat.torch as xp


class Projector():
    def __init__(self,
                 voxel_shape: float = (1.953125, 1.953125, 2),
                 volume_shape: tuple = (64, 256, 256),
                 num_rings: int = 24,
                 max_ring_difference: int = 22,
                 diameter: float = 700,
                 num_sides: int = 300,
                 num_lor_endpoints_per_side: int = 1,
                 lor_spacing: int = 1,
                 lor_radial_trim: int = 30,
                 use_res_model: bool = True,
                 fwhm: float = 4.5,
                 use_tof: bool = True,
                 num_tofbins: int=13,
                 tofbin_width: float = 60,
                 sigma_tof: float = 24.50529087048832, 
                 num_sigmas: float = 2.0, 
                 tofcenter_offset: float = 0,
                 device='cuda') -> None:
        """_summary_

        Args:
            resolution (tuple, optional): Dimensions of voxel in mm. Defaults to (3.90625, 3.90625, 6).
            volume_shape (tuple, optional): Dimensions of volume to project. Defaults to (16, 128, 128).
            num_rings (int, optional): Number of detector rings. Defaults to 16.
            diameter (float, optional): Diameter of detector rings. Defaults to 800.
            num_sides (int, optional): Number of sides . Defaults to 300.
            num_lor_endpoints_per_side (int, optional): Number of lor endpoints on one side. Defaults to 1.
            lor_spacing (int, optional): Spacing between detectors on each side. Defaults to 3.
            lor_radial_trim (int, optional): Minimal number of detectors between lor endpoints on ring. Defaults to 1.
            use_tof (bool, optional): Whether to use time-of-flight. Defaults to True.
            device (str, optional): Device for computations. Defaults to 'cuda'.
        """

        # define scanner geometry
        scanner = parallelproj.RegularPolygonPETScannerGeometry(
            xp,
            device,
            radius=diameter//2,
            num_sides=num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            ring_positions=xp.linspace(-(volume_shape[0]//2)*voxel_shape[2], (volume_shape[0]//2)*voxel_shape[2], num_rings),
            symmetry_axis=0, # when volume is (D, H, W)
        )

        # lor descriptor
        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=lor_radial_trim,
            max_ring_difference=max_ring_difference,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )

        # projector
        proj = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=volume_shape, voxel_size=voxel_shape
        )

        if use_tof:
            proj.tof_parameters = parallelproj.TOFParameters(num_tofbins=num_tofbins,
                                                             tofbin_width=tofbin_width,
                                                             sigma_tof=sigma_tof,
                                                             num_sigmas=num_sigmas,
                                                             tofcenter_offset=tofcenter_offset)

        # image-based resolution model with given gaussian FWHM
        if use_res_model:
            res_model = parallelproj.GaussianFilterOperator(
                proj.in_shape, sigma=fwhm / (2.355 * proj.voxel_size)
            )

            self.__projector = parallelproj.CompositeLinearOperator((proj, res_model))
        else:
            self.__projector = proj

        self.__use_tof = use_tof
        self.__device = device

    @property
    def device(self) -> str:
        return self.__device

    @property
    def use_tof(self) -> bool:
        return self.__use_tof

    def transform(self, x:torch.Tensor) -> torch.Tensor:
        """Forward projection."""
        return self.__projector(x) 

    def transposed_transform(self, y:torch.Tensor) -> torch.Tensor:
        """Back projection."""
        return self.__projector.adjoint(y)
    