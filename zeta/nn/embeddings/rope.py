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
    
#example usage
d = 10
k = 0.5
a = 0.1
b = 0.9
rho = 0.01

rope_power_scaling = RoPE(d=d, k=k)
modified_basis_power_scaling = rope_power_scaling.power_scaling_basis()
print(modified_basis_power_scaling)

rope_truncated = RoPE(d=d, a=a, b=b, rho=rho)
modified_basis_truncated = rope_truncated.truncated_basis()
print(modified_basis_truncated)