import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint


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

    def evaluate(self, batch, stage=None):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)
        return {'loss': loss, 'acc': acc, 'preds': preds, 'y': y}

    def epoch_evaluate(self, step_outputs, stage=None):
        y_list = []
        preds_list = []
        loss_list = []
        acc_list = []
        for output in step_outputs:
            # print(output)
            y_list.append(output['y'])
            preds_list.append(output['preds'])
            loss_list.append(output['loss'])
            acc_list.append(output['acc'])
        # print(loss_list)
        y_cat = torch.cat(y_list)
        preds_cat = torch.cat(preds_list)
        loss_cat = torch.tensor(loss_list)
        acc_cat = torch.tensor(acc_list)
        acc = accuracy(preds_cat, y_cat)
        loss = torch.mean(loss_cat)
        macc = torch.mean(acc_cat)

        if stage:
            self.log(f'{stage}_epoch_acc', acc)
            self.log(f'{stage}_epoch_loss', loss)
            self.log(f'{stage}_epoch_macc', macc)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, 'val')

    def validation_epoch_end(self, val_step_outputs):
        self.epoch_evaluate(val_step_outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

    def test_epoch_end(self, test_step_outputs):
        self.epoch_evaluate(test_step_outputs, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                'monitor': 'metric_to_track',
            }
        }


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNIST(root='./',
                    train=True,
                    download=True,
                    transform=transform)
    mnist_test = MNIST(root='./',
                       train=False,
                       download=True,
                       transform=transform)
    #mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    mnist_train, mnist_val = random_split(dataset, [59500, 500])

    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=8)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=8)
    test_loader = DataLoader(mnist_test, batch_size=64, num_workers=8)

    mc = ModelCheckpoint(
        monitor='val_epoch_acc', 
        dirpath='./', 
        filename='mnist-{epoch:02d}-{val_epoch_acc:.2f}',
        mode='max')

    model = PerceiverClassifier(input_channels=1,
                                depth=1,
                                attn_dropout=0.,
                                ff_dropout=0.,
                                num_classes=10,
                                weight_tie_layers=True)
    trainer = pl.Trainer(gpus=1,
                         precision=32,
                         log_gpu_memory=True,
                         max_epochs=40,
                         progress_bar_refresh_rate=5,
                         benchmark=True,
                         callbacks=[mc])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    main()
