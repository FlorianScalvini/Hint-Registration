import json
import torch
import argparse
import numpy as np
import torchio as tio
import monai.networks.nets
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_lightning.loggers import TensorBoardLogger


from dataset import subjects_from_csv
from utils import normalize_to_0_1, create_new_versioned_directory, BendingEnergyLoss
from Registration import RegistrationModuleSVF, UniGradIconRegistrationWrapper


def get_reg_loss(disp_flow):
    reg_criterion = BendingEnergyLoss()
    reg_loss = reg_criterion(disp_flow)
    return reg_loss

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class StaticAtlasRegistation(pl.LightningModule):
    def __init__(self, alpha=0.01, saving_path=None, atlas=None, atlas_lbl=None):
        super().__init__()
        #self.image_size = [128, 128, 128]
        self.num_classes = 20
        #self.model = RegistrationModelSVF(model=monai.networks.nets.Unet(spatial_dims=3, in_channels=2*1, out_channels=3, channels=(8,16,32), strides=(2,2)),
        #                                  inshape=self.image_size,
        #                                  int_steps=7)

        self.image_size = [175, 175, 175]
        self.model = UniGradIconRegistrationWrapper()
        #self.model = RegistrationModelSVF(model=monai.networks.nets.Unet(spatial_dims=3, in_channels=2*1, out_channels=3, channels=(8,16,32), strides=(2,2)),
        #                                  inshape=self.image_size,
        #                                  int_steps=7)
        self.saving_path = saving_path
        self.alpha = alpha
        self.atlas = atlas
        self.atlas_lbl = atlas_lbl
        self.epoch_loss_fading = 200
        self.lmbda = 500

    def forward(self, source, target):
        return self.model(source, target)

    def on_train_start(self):
        if self.atlas is None:
            self.update_atlas(init=True)
        norm_atlas = normalize_to_0_1(self.atlas)
        self.trainer.loggers[0].experiment.add_image("Atlas Sagittal Plane", TF.rotate(norm_atlas[0, :, int(self.image_size[0] / 2), :, :], 90),
                                                     0)
        self.trainer.loggers[0].experiment.add_image("Atlas Coronal Plane", TF.rotate(norm_atlas[0, :, :, int(self.image_size[1] / 2), :], 90),
                                                     0)
        self.trainer.loggers[0].experiment.add_image("Atlas Axial Plane", TF.rotate(norm_atlas[0, :, :, :, int(self.image_size[2] / 2)], 90), 0)
        self.model.train()

    def training_step(self, x):
        image = x['image'][tio.DATA]
        labelmap = x['label'][tio.DATA].float()

        '''
        repeats = np.ones(len(image.size()))
        repeats[0] = image.size(0)
        atlas = self.atlas.repeat(tuple(repeats.astype(int)))
        atlas_lbl = self.atlas_lbl.repeat(tuple(repeats.astype(int)))


        forward_flow, backward_flow = self.forward(image, atlas)
        warped_source = self.model.warped_image(image, forward_flow)
        warped_atlas = self.model.warped_image(atlas, backward_flow)

        warped_source_seg = self.model.warped_image(labelmap, forward_flow)
        warped_atlas_seg = self.model.warped_image(atlas_lbl, backward_flow)
        '''
        loss = self.model(image, self.atlas)
        warped_source = self.model.warped_image(image, self.model.model.phi_AB_vectorfield)
        warped_atlas = self.model.warped_image(self.atlas, self.model.model.phi_BA_vectorfield)
        warped_source_seg = self.model.warped_image(labelmap, self.model.model.phi_AB_vectorfield)
        warped_atlas_seg = self.model.warped_image(self.atlas_lbl, self.model.model.phi_BA_vectorfield)


        ## loss
        sim_loss = F.mse_loss(warped_source, self.atlas) + F.mse_loss(warped_atlas, image)
        seg_loss = F.mse_loss(warped_source_seg[:, 1::, ...], self.atlas_lbl[:, 1::, ...]) + F.mse_loss(warped_atlas_seg[:, 1::, ...], labelmap[:, 1::, ...])
        loss_train = (10*sim_loss + seg_loss).float()


        #source = labelmap[:, 1::, ...]
        #target = atlas[:, 1::, ...]


        # Segmentation losses
        #loss_seg2atl = F.mse_loss(warped_labelmap[:, 1::, ...], atlas)
        #loss_atl2seg = F.mse_loss(labelmap[:, 1::, ...], warped_atlas_labelmap[:, 1::, ...])

        # Regularization term
        #reg_term = self.model.regularizer(forward_flow)

        #loss_train = loss_seg2atl + loss_atl2seg + self.lmbda * reg_term

        self.log("Global loss", loss_train, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss[0]

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_train_epoch_end(self) -> None:
        self.model.eval()
        self.update_atlas(init=False)
        norm_atlas = normalize_to_0_1(self.atlas)

        self.trainer.loggers[0].experiment.add_image("Atlas Sagittal Plane", TF.rotate(norm_atlas[0, :, int(self.image_size[0] / 2), :, :], 90),
                                                     self.current_epoch + 1)
        self.trainer.loggers[0].experiment.add_image("Atlas Coronal Plane", TF.rotate(norm_atlas[0, :, :, int(self.image_size[1] / 2), :], 90),
                                                     self.current_epoch + 1)
        self.trainer.loggers[0].experiment.add_image("Atlas Axial Plane", TF.rotate(norm_atlas[0, :, :, :, int(self.image_size[2] / 2)], 90),
                                                     self.current_epoch + 1)

        if self.saving_path is not None:
            torch.save(self.atlas, os.path.join(self.saving_path, "current_atlas.pth"))
            torch.save(self.atlas_lbl, os.path.join(self.saving_path, "current_atlas_lbl.pth"))

            o = tio.ScalarImage(tensor=self.atlas.squeeze(dim=0).cpu().detach().numpy(),
                                affine=self.trainer.train_dataloader.dataset[0]["image"].affine)
            o.save(os.path.join(self.saving_path, "atlas.nii.gz"))



    def update_atlas(self, init=False):
        dataset = self.trainer.train_dataloader.dataset
        atlas_image_update = torch.zeros(dataset[0]['image'][tio.DATA].shape).unsqueeze(dim=0).to(device)
        atlas_labelmap_update = torch.zeros(dataset[0]['label'][tio.DATA].shape).unsqueeze(dim=0).to(device)

        with torch.no_grad():
            for _, x in enumerate(dataset):
                if init is True:
                    atlas_image_update += x['image'][tio.DATA].unsqueeze(dim=0).to(device)
                    atlas_labelmap_update += x['label'][tio.DATA].unsqueeze(dim=0).to(device).float()
                else:
                    image = x['image'][tio.DATA].unsqueeze(dim=0).to(device)
                    labelmap = x['label'][tio.DATA].unsqueeze(dim=0).to(device).float()

                    #source = labelmap[:, 1::, ...]
                    #target = self.atlas_lbl[:, 1::, ...]
                    '''
                    forward_flow, _ = self.forward(image, self.atlas)
                    warped_labelmap = self.model.warped_image(labelmap, flow=forward_flow)
                    warped_image = self.model.warped_image(image, flow=forward_flow)
                    '''

                    forward_flow, _ = self.model(image, self.atlas)
                    warped_image = self.model.wrap(image, forward_flow)
                    warped_labelmap = self.model.wrap(labelmap, forward_flow)

                    atlas_image_update += warped_image
                    atlas_labelmap_update += warped_labelmap

        atlas_image_update /= len(dataset)
        atlas_labelmap_update /= len(dataset)

        if init is True:
            self.atlas = atlas_image_update
            self.atlas_lbl = atlas_labelmap_update
        else:
            self.atlas = (self.atlas * (1.0 - self.alpha) + atlas_image_update * self.alpha)
            self.atlas_lbl = (self.atlas_lbl * (1.0 - self.alpha) + atlas_labelmap_update * self.alpha)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Static Atlas for 3D Image with pretrained network')
    parser.add_argument('-t', '--csv_file', help='csv file ', type=str, required=False, default="./dataset.csv")
    parser.add_argument('-c', '--config', help='Config file', type=str, required=False, default="./config.json")
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default=200)
    parser.add_argument('-s', '--save', help='Saving path', type=str, required=False, default="./save/")
    args = parser.parse_args()

    json_loader = json.load(open(args.config))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(175),
        tio.OneHot(20)
    ]

    subjects = subjects_from_csv(args.csv_file, age=False)
    subjects_dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(train_transform))

    train_loader = tio.SubjectsLoader(
        dataset=subjects_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    saving_path = create_new_versioned_directory(os.path.join(args.save, "version"))
    tb_logger = TensorBoardLogger("./save/")

    trainer_reg = pl.Trainer(
        max_epochs=args.epochs,
        logger=tb_logger)

    model = StaticAtlasRegistation(saving_path=saving_path)
    trainer_reg.fit(model, train_dataloaders=train_loader)
    torch.save(model.state_dict(), os.path.join(saving_path, "model.pth"))

