import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet2


class FunctionFromVectorField(RegistrationModule):
    """
    Wrap an inner neural network 'net' that returns a tensor of displacements
    [B x N x H x W (x D)], into a RegistrationModule that returns a function that
    transforms a tensor of coordinates
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        tensor_of_displacements = self.net(image_A, image_B)
        displacement_field = self.as_function(tensor_of_displacements)

        def transform(coordinates):
            if hasattr(coordinates, "isIdentity") and coordinates.shape == tensor_of_displacements.shape:
                return coordinates + tensor_of_displacements
            return coordinates + displacement_field(coordinates)

        return transform

def make_network(input_shape, include_last_step=False):
    net = FunctionFromVectorField(UNet2(5, [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]]))

    for _ in range(2):
        net = TwoStepRegistration(
            DownsampleRegistration(net),
            FunctionFromVectorField(UNet2(5, [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]]))
        )
    if include_last_step:
        net = TwoStepRegistration(net, FunctionFromVectorField(UNet2(5, [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]]))

    return net


class TwoStepRegistration(nn.Module):
    """Combine two Modules.

    First netPhi is called on the input images, then image_A is warped with
    the resulting field, and then netPsi is called on warped A and image_B
    in order to find a residual warping. Finally, the composition of the two
    transforms is returned.
    """

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):
        phi = self.netPhi(image_A, image_B)
        psi = self.netPsi(image_A, phi(self.identity_map),
            image_B
        )
        return lambda tensor_of_coordinates: phi(psi(tensor_of_coordinates))

class DownsampleRegistration(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.avg_pool = F.avg_pool3d
        self.interpolate_mode = "trilinear"
        self.downscale_factor = 2

    def forward(self, image_A, image_B):

        image_A = self.avg_pool(image_A, 2, ceil_mode=True)
        image_B = self.avg_pool(image_B, 2, ceil_mode=True)
        return self.net(image_A, image_B)

def _make_network():
    unet_1 = UNet2(5, [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]])
    down_1 = DownsampleRegistration(net=unet_1)

    unet_2 = UNet2(5, [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]])
    two_1 = TwoStepRegistration(down_1, unet_2)
    down_2 = DownsampleRegistration(two_1)

    unet_3 = UNet2(5, [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]])
    return TwoStepRegistration(down_2, unet_3)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class UniGradIcon(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _make_network()
        self.spatialTrans
    def forward(self, image_A, image_B):
        return self.net(image_A, image_B)

    def getLoss(self, image_A, image_B):
        assert identity_map.shape[2:] == image_A.shape[2:]
        assert identity_map.shape[2:] == image_B.shape[2:]

        identity_map.isIdentity = True

        phi_AB = self.net(image_A, image_B)
        phi_BA = self.net(image_B, image_A)

        phi_AB_vectorfield = phi_AB(identity_map)
        phi_BA_vectorfield = phi_BA(identity_map)

        warped_image_A = SpatialTransformer

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return icon.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon.losses.flips(self.phi_BA_vectorfield),
        )


