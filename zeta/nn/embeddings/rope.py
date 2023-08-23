#from paper:: https://arxiv.org/pdf/2308.10882.pdf

import torch 

class RoPE:
    def __init__(
            self,
            d,
            k=None,
            a=None,
            b=None,
            rho=None
    ):
        self.d = d
        self.k = k
        self.a = a
        
        self.b = b
        self.rho = rho
    
    def power_scaling_basis(self):
        basis = torch.pow(10000 - 2 * torch.arange(self.d)[:, None] / self.d, 2 * 
                          torch.arange(self.d)[None, :] / self.d)
        modified_basis = (1 - 2 * torch.arange(self.d)[:, None] / 
                          self.d).pow(self.k) * basis
        return modified_basis
    
    def truncated_basis(self):
        basis = torch.pow(10000 - 2 * torch.arange(self.d)[:, None] / self.d, 2 * 
                          torch.arange(self.d)[None, :] / self.d)
        modified_basis = torch.where(basis >= self.b, basis, torch.where(basis > self.a, self.rho, 0))
        return modified_basis
    

