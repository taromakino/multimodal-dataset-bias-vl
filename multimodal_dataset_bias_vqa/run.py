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
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    model = FIBERTransformerSS(_config)

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode=_config["val_mode"],
        save_last=True,
        filename="best"
    )
    callbacks = [checkpoint_callback]

    task = f'{_config["task"]}'
    logger = pl.loggers.CSVLogger(
        _config["log_dir"],
        name="",
        version=_config["seed"],
    )

    model.freeze()
    if task == "vae":
        model.vqa_classifier.requires_grad_(True)
    elif task == "posterior_kld":
        model.vqa_classifier.encoder_x.requires_grad_(True)

    num_gpus = _config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"])
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"], # Load everything (model weights, optimizer, lr scheduler, etc)
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
