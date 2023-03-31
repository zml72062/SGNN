

from torch_geometric.datasets import GNNBenchmarkDataset
from models.input_encoder import LinearEncoder
from models.GNNs import *
import train_utils
import pytorch_lightning as pl
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torchmetrics
import wandb

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"




def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="CSL", help='name of dataset')
    parser.add_argument('--folds', type=int, default=10, help='number of repeat run')
    args = parser.parse_args()
    args = train_utils.update_args(args)


    path, pre_transform, follow_batch = train_utils.data_setup(args)

    dataset = GNNBenchmarkDataset(path,
                         name=args.dataset_name,
                         pre_transform=pre_transform,
                         transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))
    args.out_channels = dataset.num_classes

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*train_utils.k_fold(dataset, args.folds, args.seed))):

        # Set random seed
        seed = train_utils.get_seed(args.seed)
        pl.seed_everything(seed)

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        logger = WandbLogger(name=f'fold_{str(fold+1)}', project=args.exp_name, log_model=True, save_dir=args.save_dir)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))


        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              follow_batch=follow_batch)
        loss_cri = nn.CrossEntropyLoss()
        evaluator = torchmetrics.classification.MulticlassAccuracy(num_classes=dataset.num_classes)
        init_encoder = LinearEncoder(dataset.num_features, args.hidden_channels)


        modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                           evaluator=evaluator,
                                           args=args,
                                           init_encoder=init_encoder)
        trainer = Trainer(
                        accelerator="auto",
                        devices="auto",
                        max_epochs=args.num_epochs,
                        enable_checkpointing=True,
                        enable_progress_bar=True,
                        logger=logger,
                        callbacks=[
                            TQDMProgressBar(refresh_rate=20),
                            ModelCheckpoint(monitor="val/metric", mode="max"),
                            LearningRateMonitor(logging_interval="epoch"),
                            timer
                        ]
                        )


        trainer.fit(modelmodule, datamodule)
        modelmodule.set_test_eval_still()
        val_result, test_result = trainer.validate(modelmodule, datamodule, ckpt_path="best")
        results = {"final/best_val_metric": val_result["val/metric"],
                   "final/best_test_metric": test_result["test/metric"],
                   "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,

                   }
        logger.log_metrics(results)
        wandb.finish()

    return


if __name__ == "__main__":
    main()
