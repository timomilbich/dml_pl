import wandb
import numpy as np
from tqdm import tqdm

num_classes_to_dataset = {
    98: 'cars196',
    100: 'cub200',
    11318: 'sop',
}

configentry_to_dataset = {
    'data.CARS196.DATA': 'cars196',
    'data.CUB200.DATA': 'cub200',
    'data.SOP.DATA': 'sop',
}

def get_wandb_data():
    api = wandb.Api()
    entity, project = "timomil", "DML_PL"  # set to your entity and project
    runs = api.runs(entity + "/" + project)

    # read weights and biases runs (total)
    summary_list, config_list, name_list, group_list, data_list = [], [], [], [], []
    for run in tqdm(runs, desc='Collect wandb runs data'):

        # only consider finised runs (not 'running', 'crashed', 'failed', etc.)
        if run.state != "finished":
            continue

        try:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            run_summary = run.summary._json_dict

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')}

            run_data = dict()
            ### get recall@1 per epoch
            run_hist = run.history()
            id_tmp = (run_hist.columns == 'val/e_recall@1').nonzero()[0][0]
            recallone_vals = run_hist.values[:, id_tmp]
            run_data['r@1_max'] = np.max(np.nan_to_num(recallone_vals))

            ### get general run info
            run_data['arch'] = run_config['params/config/Architecture/params/arch'] if 'params/config/Architecture/params/arch' in run_config.keys()\
                else run_config['model/params/config/Architecture/params/arch']
            run_data['loss_function'] = run_config['params/config/Loss/params/name'] if 'params/config/Loss/params/name'in run_config.keys()\
                else run_config['model/params/config/Loss/params/name']
            run_data['embed_dims'] = run_config['params/config/Architecture/params/embed_dim'] if 'params/config/Architecture/params/embed_dim' in run_config.keys()\
                else run_config['model/params/config/Architecture/params/embed_dim']
            if run_data['embed_dims'] == -1:
                run_data['embed_dims'] = 384

            ### get dataset
            if 'data/params/train/target' in run_config.keys():
                run_data['dataset'] = configentry_to_dataset[run_config['data/params/train/target']]
            else:  # legacy
                run_data['dataset'] = num_classes_to_dataset[run_config['params/config/Loss/params/n_classes']]

            ### get ooDML split
            if 'data/params/validation/params/ooDML_split_id' in run_config.keys():
                run_data['ooDML_split_id'] = run_config['data/params/validation/params/ooDML_split_id']
            else:  # legacy
                if 'ooDML' in run.group:
                    run_data['ooDML_split_id'] = int(run.name[-1])
                else:
                    run_data['ooDML_split_id'] = -1

        except:
            print(f'Omitting run [{run.name}] in group [{run.group}]')
            continue

        name_list.append(run.name)
        group_list.append(run.group)
        summary_list.append(run_summary)
        config_list.append(run_config)
        data_list.append(run_data)

    return name_list, group_list, summary_list, config_list, data_list