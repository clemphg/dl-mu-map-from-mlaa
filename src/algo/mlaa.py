

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch


class MLAA():

    def __init__(self,
                 projector,
                 tof_projector = None,
                 is_tof: bool = False) -> None:

        self.projector = projector

        if is_tof and tof_projector is None:
            raise AttributeError("Please input ToF projector to use ToF")
        
        self.tof_projector = tof_projector
        self.is_tof = is_tof

    def solve(self,
              y: torch.Tensor,
              mu_init: torch.Tensor,
              tau: float,
              bckg_pet: float,
              n_iter: int,
              display=False,
              eps:float = 1e-7):
        
        # Initialization
        nb_slices = mu_init.shape[0]
        dim_img = mu_init.shape[1]

        x = torch.ones((nb_slices, dim_img, dim_img), dtype=torch.float32, device=y.device)
        mu = mu_init.to(y.device)
        
        # Compute "blank scan"
        A1 = self.A(torch.ones((nb_slices, dim_img, dim_img), dtype=torch.float32, device=y.device))
        
        # Handle TOF case
        if self.is_tof:
            y_noTOF = torch.sum(y, dim=-1)  # summing along the TOF dimension
        else:
            y_noTOF = y

        
        for k in tqdm(range(n_iter), total=n_iter, ncols=100):

            # MLEM update for x
            attn = torch.exp(-self.A(mu))
            if self.is_tof:
                r_attn = attn.unsqueeze(-1).repeat(1, 1, 1, y.shape[-1])
            else:
                r_attn = attn
            p = self.PT(r_attn)
            x = x / (tau * p) * self.PT(r_attn * y / (r_attn * self.P(x) + bckg_pet))

            # set nans to zero
            x[torch.isnan(x)] = 0

            # MLTR update for mu
            Px = self.P(x)
            if self.is_tof:
                psi = attn * torch.sum(Px, dim=-1) # sum over extra dim
            else:
                psi = attn * Px

            s = bckg_pet
            grad = self.AT( psi / (psi + s) * (tau * (psi + s) - y_noTOF))
            hess = self.AT(A1 * tau * (psi)**2 / (psi + s))
            mu = mu + grad / hess

            # set nans to zero and clip
            mu[torch.isnan(mu)] = 0
            mu = torch.clamp(mu, min=0)
            
            # could be removed
            if display:
                fig, axs = plt.subplots(1, 4, figsize=(10,2))
                p0 = axs[0].imshow(x.cpu().numpy()[8], cmap='hot')
                plt.colorbar(p0, ax=axs[0])
                axs[0].axis('off')
                axs[0].set_title(f"MLAA $\\lambda$, iteration {k+1}")
                
                p1 = axs[1].imshow(mu.cpu().numpy()[8], cmap='bone')
                plt.colorbar(p1, ax=axs[1])
                axs[1].axis('off')
                axs[1].set_title(f"MLAA $\\mu$, iteration {k+1}")
                
                p2 = axs[2].imshow(x.cpu().numpy()[:, 64], cmap='hot')
                plt.colorbar(p2, ax=axs[2])
                axs[2].axis('off')
                axs[2].set_title(f"MLAA $\\lambda$, iteration {k+1}")
                
                p3 = axs[3].imshow(mu.cpu().numpy()[:, 64], cmap='bone')
                plt.colorbar(p3, ax=axs[3])
                axs[3].axis('off')
                axs[3].set_title(f"MLAA $\\mu$, iteration {k+1}")
                plt.tight_layout()
                plt.show()
        
        return x, mu
    
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