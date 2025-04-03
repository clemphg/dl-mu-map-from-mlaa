

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


class MLEM():

    def __init__(self,
                 projector,
                 tof_projector) -> None:
        super().__init__()

        self.projector = projector
        self.tof_projector = tof_projector
        self.is_tof = False

    def solve(self, 
              y: torch.Tensor, 
              mu: torch.Tensor,
              tau: float=1.0,
              bckg_pet: float=0,
              n_iter: int=100,
              eps: float=1e-7, 
              use_tof: bool=False,
              display=False):
        
        self.is_tof = use_tof

        # compute attenuation sinogram
        x_att_fwd = self.projector.transform(mu)
        att_sino = torch.exp(-x_att_fwd)
        if use_tof:
            att_sino = att_sino.unsqueeze(-1).repeat(1, 1, 1, y.shape[-1])

        # initialisation
        x = torch.ones(self.projector.volume_shape, dtype=torch.float32, device=y.device)
        norm1 = self.PT(torch.ones_like(y))
        
        # updates
        for q in tqdm(range(n_iter)):
            projx = self.P(x)
            update_ratio = y / (tau * (att_sino * projx + bckg_pet + eps))
            x = x * (self.PT(update_ratio) / norm1)
            
            if display and q%10==0:
                # plt.imshow(x[x.shape[0]//2].cpu().numpy())
                # plt.title(f'MLEM iteration {q+1}')
                # plt.colorbar()
                # plt.axis('off')
                # plt.show()

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                p0 = axs[0].imshow(x[x.shape[0]//2].cpu().numpy())
                plt.colorbar(p0, ax=axs[0])
                axs[0].set_title('x')

                p1 = axs[1].imshow(norm1[norm1.shape[0]//2].cpu().numpy())
                plt.colorbar(p1, ax=axs[1])
                axs[1].set_title('norm')

                # p2 = axs[2].imshow(update_ratio[:, :, 100].squeeze().cpu().numpy())
                # plt.colorbar(p2, ax=axs[2])
                # axs[2].set_title('ratio')

                plt.tight_layout()
                plt.show()

        x = torch.nan_to_num(x)

        return x
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector.transform(x)
    
    def AT(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector.transposed_transform(x)
    
    def P(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tof:
            res = self.tof_projector.transform(x)
        else:
            res = self.projector.transform(x)
        return res
    
    def PT(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tof:
            res = self.tof_projector.transposed_transform(x)
        else:
            res = self.projector.transposed_transform(x)
        return res