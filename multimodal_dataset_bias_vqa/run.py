import copy
import os
import pytorch_lightning as pl
import resource
from config import ex
from modules.fiber_module import FIBERTransformerSS
from datamodules.multitask_datamodule import MTDataModule


os.environ["NCCL_DEBUG"] = "INFO"


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def main(config):
    config = copy.deepcopy(config)
    pl.seed_everything(config["seed"])

    dm = MTDataModule(config, dist=True)
    model = FIBERTransformerSS(config)

    os.makedirs(config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode=config["val_mode"],
        save_last=True,
        filename="best"
    )
    callbacks = [checkpoint_callback]

    task = f'{config["task"]}'
    logger = pl.loggers.CSVLogger(
        config["log_dir"],
        name="",
        version=config["seed"],
    )

    model.freeze()
    if task == "vae":
        model.vqa_classifier.requires_grad_(True)
    elif task == "posterior_kld":
        model.vqa_classifier.encoder_x.requires_grad_(True)

    num_gpus = config["num_gpus"] if isinstance(config["num_gpus"], int) else len(config["num_gpus"])
    grad_steps = max(config["batch_size"] // (config["per_gpu_batchsize"] * num_gpus * config["num_nodes"]), 1)

    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        resume_from_checkpoint=config["resume_from"], # Load everything (model weights, optimizer, lr scheduler, etc)
    )

    if not config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
