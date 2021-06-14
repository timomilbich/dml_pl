import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from utils.auxiliaries import instantiate_from_config, nondefault_trainer_args
from utils.callbacks import SetupCallback
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
import torchvision
import torch


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="/export/home/tmilbich/PycharmProjects/dml_pl/experiments",
        nargs="?",
        help="postfix for logdir",
    )
    # default = "/home/timomil/PycharmProjects/Codebase/experiments",
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="MNIST.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", 
        "--project", 
        default="test",
        help="name of new or path to existing project",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )

    return parser

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has some defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser_kwargs_set = [
    # "--gpus", "1,",
    "--base", "configs/cub200.yaml",
    "--project" , "test",
    "--debug", "True",
    # "--overfit_batches", "10",
    # "--limit_train_batches", "0.2",
    # "--limit_val_batches", "0.2",
    # "--track_grad_norm", "2",
    # "--fast_dev_run", "3", #runs n train, val and test batches
    ]

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args(parser_kwargs_set)
    print(opt.gpus)

    ## Setup GPU's
    GPUs = opt.gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].find("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")-1]

    else:
        name = opt.name if opt.name else ""
        nowname = name + '/' + opt.project + "_" + now
        logdir = os.path.join(nowname, "logs")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # default to ddp
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        trainer_config['gpus'] = ",".join([str(i) for i in range(len(trainer_config['gpus'].strip(",").split(',')))])
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # Initialize model from config
        model = instantiate_from_config(config.model)

        ##
        trainer_kwargs = dict()

        ## Set up Wandb for logging
        logger_cfg = lightning_config.logger or OmegaConf.create()
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        _ = os.system('wandb login {}'.format(lightning_config.logger['params']['wandb_key']))
        os.environ['WANDB_API_KEY'] = lightning_config.logger['params']['wandb_key']
        wandb_logger = WandbLogger(opt.project, logdir, project=lightning_config.logger['params']['project'], offline=opt.debug, version=opt.project,
                                   group=lightning_config.logger['params']['group'])
        wandb_logger.log_hyperparams(config.model)
        trainer_kwargs["logger"] = wandb_logger

        # Setup saving modelcheckpoint
        # lightning_config.modelcheckpoint['params']['dirpath'] = os.path.join(ckptdir, "{epoch:06}")
        lightning_config.modelcheckpoint['params']['dirpath'] = ckptdir
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(lightning_config.modelcheckpoint)

        # Define callbacks -Setupcallback is always called
        setup_callback = SetupCallback(opt.resume, now, logdir, ckptdir, cfgdir, config, lightning_config)
        if lightning_config.callbacks is not None:
            logging_callbacks = [instantiate_from_config(lightning_config.callbacks[k]) for k in lightning_config.callbacks]
        else:
            logging_callbacks = []
        trainer_kwargs["callbacks"] = [setup_callback, *logging_callbacks]

        ## Define Trainer
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        # data.prepare_data()
        data.setup()

        # configure learning rate
        # bs, base_lr, ngpu = config.data.params.batch_size, config.model.base_learning_rate, len(
        #     GPUs.strip(",").split(','))
        # accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        # model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        # print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
        #     model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        weight_decay, gamma, scheduler, tau = config.model.weight_decay, config.model.gamma, config.model.scheduler, config.model.tau
        model.weight_decay = weight_decay
        model.gamma = gamma
        model.tau = tau

        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        model.learning_rate = 1 * 1 * 1 * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, 0, bs, base_lr))
        print(f"Other optimizer paramters set to: \nweight_decay: {weight_decay}\nscheduler: {scheduler}\ngamma: {gamma}\ntau: {tau}")

        # run
        if opt.train:
            trainer.tune(model, data)
            trainer.fit(model, data)
        # if not opt.no_test and not trainer.interrupted:
        #     # TODO provide option to change test_step of model
        #     trainer.test(model, data)
    except Exception:
        # move newly created debug project to debug_runs
        raise
        if opt.debug:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            try:
                os.rename(logdir, dst)
            except:
                pass