import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import heads, roberta, swin_transformer
from modules.roberta import RobertaModel
from modules.swin_helpers import swin_adapt_position_encoding
from modules.vae import Vae
from pytorch_lightning.metrics import Metric


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1] # Column with max logit value
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1) # Array of zeros, except for one at the column with max logit value
        scores = one_hots * target # Array of zeros, except for the target at the column with max logit value

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total # Average target at the column with max logit value


class FIBERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        resolution_after = config["image_size"]
        self.num_fuse_block = config["num_fuse_block"]
        self.num_text_layer = config["num_layers"]
        roberta.NUM_FUSE_BLOCK = swin_transformer.NUM_FUSE_BLOCK = self.num_fuse_block
        roberta.DIM_IMG = config["input_image_embed_size"]
        swin_transformer.DIM_TXT = config["input_text_embed_size"]

        self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])

        self.cross_modal_text_transform_itc = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform_itc = nn.Linear(config["input_image_embed_size"], config["hidden_size"])

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                getattr(swin_transformer, config["vit"])(
                    pretrained=config["pretrained_vit"],
                    config=config,
                )
                RobertaModel.from_pretrained(config["tokenizer"])

            torch.distributed.barrier()

        self.vit_model = getattr(swin_transformer, config["vit"])(
            pretrained=config["pretrained_vit"],
            config=config,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_transformer = RobertaModel.from_pretrained(config["tokenizer"])

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.itc_pooler = config["itc_pooler"]
        if self.itc_pooler:
            self.cross_modal_image_pooler_itc = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler_itc = heads.Pooler(config["hidden_size"])

        self.vae = Vae(config["hidden_size"], config["hidden_dims"], config["latent_size"], config["vqav2_label_size"],
            config["n_components"], config["n_samples"])
        self.vqa_score = VQAScore()

        exclude_keys = ["image_queue", "text_queue", "queue_ptr", "queue_total", "image_input_queue", "text_input_queue",
            "text_input_mask_queue"]
        if config["load_path"] != "":
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            for key in exclude_keys:
                if key in state_dict:
                    state_dict.pop(key)
            if os.path.basename(config["load_path"]) == "fiber_pretrain.ckpt" and not config["test_only"]:
                state_dict = swin_adapt_position_encoding(
                    state_dict, before=config["resolution_before"], after=resolution_after
                )
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        text_only=False,
        image_only=False,
    ):
        if not text_only:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch[imgkey][0]

        if not image_only:
            do_mlm = "_mlm" if mask_text else ""
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]

        image_embeds = self.vit_model.patch_embed(img)
        if self.vit_model.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.vit_model.absolute_pos_embed
        image_embeds = self.vit_model.pos_drop(image_embeds)
        for layer_i, layer in enumerate(self.vit_model.layers[:2]):
            image_embeds = layer(image_embeds)

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        num_pre_text = self.num_text_layer - self.num_fuse_block
        for layer_i, layer in enumerate(self.text_transformer.encoder.layer[:num_pre_text]):
            text_embeds = layer(text_embeds, extend_text_masks)[0]

        num_pre_block = 8 + num_pre_text
        for blk_cnt, blk in enumerate(self.vit_model.layers[2].blocks):
            if blk_cnt < num_pre_block:
                image_embeds = blk(image_embeds)
            else:
                fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
                text_embeds = self.text_transformer.encoder.layer[blk_cnt - 8](
                    text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds)
                )[0]
                image_embeds = fuse_image_embeds

        if self.vit_model.layers[2].downsample is not None:
            image_embeds = self.vit_model.layers[2].downsample(image_embeds)

        for blk_cnt, blk in enumerate(self.vit_model.layers[3].blocks):
            fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
            text_embeds = self.text_transformer.encoder.layer[blk_cnt + 10](
                text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds), last_norm=(blk_cnt == 0)
            )[0]
            image_embeds = fuse_image_embeds

        if self.vit_model.layers[3].downsample is not None:
            image_embeds = self.vit_model.layers[3].downsample(image_embeds)

        text_embeds = self.cross_modal_text_transform(text_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)

        cls_feats_text = self.cross_modal_text_pooler(text_embeds)
        avg_image_feats = self.avgpool(image_embeds.transpose(1, 2)).view(image_embeds.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_embeds,
            "image_feats": image_embeds,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image": img,
        }

        return ret


    def make_vqa_targets(self, batch):
        vqa_labels = batch["vqa_labels"]
        vqa_scores = batch["vqa_scores"]
        vqa_targets = torch.zeros(len(vqa_labels), self.config["vqav2_label_size"]).to(self.device)

        for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(_label, _score):
                vqa_targets[i, l] = s

        return vqa_targets


    def forward(self, batch):
        infer = self.infer(batch)
        x = infer["cls_feats"]
        y = self.make_vqa_targets(batch)
        out = self.vae(x, y)
        self.vqa_score(out.pop("logits_y_xz"), y)
        return out


    def training_step(self, batch, batch_idx):
        out = self(batch)
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("val_kld", out["kld"], on_step=False, on_epoch=True)
        
        
    def validation_epoch_end(self, outs):
        self.log("val_score", self.vqa_score.compute())
        self.vqa_score.reset()


    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.log("test_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("test_kld", out["kld"], on_step=False, on_epoch=True)


    def test_epoch_end(self, outs):
        self.log("test_score", self.vqa_score.compute())
        self.vqa_score.reset()


    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.config["learning_rate"])