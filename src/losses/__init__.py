from .inverse_consistency import InverseConsistency, IconInverseConsistency, GradIconInverseConsistency
from .regularization import MagnitudeLoss, Grad3d
from .jacobian import Jacobianloss, calculate_jacobian
from .utils_loss import GetLoss, PairwiseRegistrationLoss
from .similarity import NCCLoss