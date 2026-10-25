from config import ConfigManager


def test_base_defaults_match_paper():
    cfg = ConfigManager().load_config()
    T, L, M, D = cfg.training, cfg.loss, cfg.model, cfg.data
    assert (T.num_epochs, T.prototype_warmup_epochs) == (200, 10)
    assert (T.encoder_lr, T.prototype_lr, T.prototype_step_size) == (1e-5, 0.1, 15)
    assert (L.fpt_weight, L.bht_weight, L.center_weight, L.norm_weight, T.brm_lambda) == (1.0, 1.0, 0.5, 0.005, 0.4)
    assert (L.fpt_margin, L.fpt_similarity_weight_strength, L.bht_margin) == (0.3, 2.0, 0.3)
    assert (L.center_similarity_weight_strength, L.center_variance_weight_strength) == (1.0, 2.5)
    assert (T.brm_tau, T.mixup_alpha, T.brm_margin, T.brm_beta, T.sep_gamma, T.sep_ema_decay) == (0.7, 0.4, 0.5, 0.7, 1.5, 0.9)
    assert M.ema_decay == 0.999 and M.prototype_init_gain == 0.5
    assert D.train_batch_size == 32 and D.num_instance == 4 and D.sampler == "pk"


def test_named_configs_and_overrides():
    cm = ConfigManager()
    cfg = cm.load_config(config_name=["veri", "dinov2_vehicle"], cli_args=["--system.seed", "7", "--training.brm_lambda=0"])
    assert cfg.data.dataset_name == "veri"
    assert cfg.model.backbone == "dinov2" and tuple(cfg.data.size_train) == (196, 392)
    assert cfg.system.seed == 7 and cfg.training.brm_lambda == 0


def test_every_config_loads():
    cm = ConfigManager()
    for name in cm.list_available_configs():
        cm.load_config(config_name=[name])


def test_print_configuration_runs(capsys):
    from training.config_utils import print_configuration

    print_configuration(ConfigManager().load_config(config_name=["vehicleid"]))
    out = capsys.readouterr().out
    assert "vehicleid" in out and "BRM" in out
