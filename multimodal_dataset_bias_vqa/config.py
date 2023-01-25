from sacred import Experiment


ex = Experiment("FIBER")


@ex.config
def config():
    task = None
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    val_mode = "max"
    batch_size = (
        4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    )

    # Image settings
    train_transform_keys = ["albef"]
    val_transform_keys = ["albef"]
    image_size = 384
    vit = "swin_base_patch4_window12_384_in22k"
    image_only = False
    draw_false_image = 0
    input_image_embed_size = 1024
    resolution_before = 384
    pretrained_vit = True

    # Text settings
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "roberta-base"
    vocab_size = 50265
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    input_text_embed_size = 768

    # Transformer settings
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    num_fuse_block = 6
    itc_pooler = True  # does not make a difference

    # Optimizer settings
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream settings
    get_recall_metric = False
    get_recall_metric_itc = True
    cider_path = None

    # PL Trainer settings
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 4
    num_nodes = 1
    load_path = ""
    num_workers = 20
    precision = 32

    # VAE settings
    latent_size = 512
    n_samples = 512
    n_posteriors = 512
    test_posteriors_path = None

    # VQA settings
    is_cp = False
    is_analysis = False


@ex.named_config
def task_vae():
    task = "vae"
    datasets = ["vqa"]
    val_check_interval = 1.0
    batch_size = 512
    max_epoch = 100
    warmup_steps = 0.1
    learning_rate = 1e-3
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_posterior_kld():
    task = "posterior_kld"
    datasets = ["vqa"]
    val_mode = "min"
    val_check_interval = 1.0
    batch_size = 512
    max_epoch = 100
    warmup_steps = 0.1
    learning_rate = 1e-3
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False


@ex.named_config
def task_backdoor_adjustment():
    task = "backdoor_adjustment"
    datasets = ["vqa"]
    val_check_interval = 1.0
    batch_size = 512
    max_epoch = 1
    warmup_steps = 0.1
    learning_rate = 0.001
    lr_mult_cross_modal = 5
    lr_mult_head = 50
    max_text_len = 50
    train_transform_keys = ["albef_randaug"]
    val_transform_keys = ["albef"]
    image_size = 576
    pretrained_vit = False
    per_gpu_batchsize = 1
    test_only = True