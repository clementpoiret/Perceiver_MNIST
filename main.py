import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from utils.perceiver import Perceiver


class PerceiverClassifier(pl.LightningModule):

    def __init__(self,
                 input_channels=3,
                 input_axis=2,
                 num_freq_bands=6,
                 max_freq=10.,
                 depth=6,
                 num_latents=256,
                 latent_dim=512,
                 cross_heads=1,
                 latent_heads=8,
                 cross_dim_head=64,
                 latent_dim_head=64,
                 num_classes=1000,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 weight_tie_layers=False,
                 fourier_encode_data=True,
                 self_per_cross_attn=2):
        super().__init__()
        self.save_hyperparameters()

        self.perceiver = Perceiver(input_channels=input_channels,
                                   input_axis=input_axis,
                                   num_freq_bands=num_freq_bands,
                                   max_freq=max_freq,
                                   depth=depth,
                                   num_latents=num_latents,
                                   latent_dim=latent_dim,
                                   cross_heads=cross_heads,
                                   latent_heads=latent_heads,
                                   cross_dim_head=cross_dim_head,
                                   latent_dim_head=latent_dim_head,
                                   num_classes=num_classes,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   weight_tie_layers=weight_tie_layers,
                                   fourier_encode_data=fourier_encode_data,
                                   self_per_cross_attn=self_per_cross_attn)

    def forward(self, x):
        return self.perceiver(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("Training Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("Validation Loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('Test Loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


def main():
    dataset = MNIST(root='./',
                    train=True,
                    download=True,
                    transform=transforms.ToTensor())
    mnist_test = MNIST(root='./',
                       train=False,
                       download=True,
                       transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=16, num_workers=24)
    val_loader = DataLoader(mnist_val, batch_size=16, num_workers=24)
    test_loader = DataLoader(mnist_test, batch_size=16, num_workers=24)

    model = PerceiverClassifier(input_channels=1,
                                num_classes=10,
                                weight_tie_layers=True)

    trainer = pl.Trainer(gpus=1,
                         precision=32,
                         log_gpu_memory=True,
                         max_epochs=32,
                         progress_bar_refresh_rate=5,
                         benchmark=True,
                         checkpoint_callback=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    main()
