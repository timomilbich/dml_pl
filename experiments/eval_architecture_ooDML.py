import experiments

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# lookups
default_split = {
    'cub200': 2,
    'cars196': 0,
    'sop': 0,
}

# get wandb run data
name_list, group_list, summary_list, config_list, data_list = experiments.get_wandb_data()

project_group = 'ooDML_msloss'
labelspace = 'arch'
dataset = 'cub200' # 'cub200'

# filter runs data - project group
ids_valid = [i for i, group in enumerate(group_list) if group == project_group]
name_list = [name_list[i] for i in ids_valid]
data_list = [data_list[i] for i in ids_valid]

# filter runs data - dataset
ids_valid = [i for i, d in enumerate(data_list) if d['dataset'] == dataset]
name_list = [name_list[i] for i in ids_valid]
data_list = [data_list[i] for i in ids_valid]

# get labels
labels_list = [d[labelspace] for d in data_list]
labels_unique = sorted(list(set(labels_list)))

# gather ooDML progressions
recall_seqs = {label: [] for label in labels_unique}
split_seqs = {label: [] for label in labels_unique}

for data in data_list:
    label_tmp = data[labelspace]
    recall_seqs[label_tmp].append(data['r@1_max'])
    split_seqs[label_tmp].append(data['ooDML_split_id'])

# sort progressions by split id
n_x_ticks = -1
for label in labels_unique:
    recs = recall_seqs[label]
    split_ids = split_seqs[label]

    # sort
    recs = [x for _, x in sorted(zip(split_ids, recs))]

    # reassign
    recall_seqs[label] = recs
    split_seqs[label] = sorted(split_ids)

    if n_x_ticks == -1:
        n_x_ticks = len(recs)

# plotting
x_label = 'split_id'
x_ticks = list(range(1, n_x_ticks + 1)) # e.g. split ids
colors = ['green', 'red', 'blue', 'magenta']
linestyles = ['solid', 'solid', 'solid', 'solid'] # 'dashed'
alphas = [1.0, 1.0, 1.0, 1.0, 1.0]


fig, ax = plt.subplots(figsize=(5, 4))
for i, label in enumerate(labels_unique):
    ax.plot(x_ticks, recall_seqs[label], label=f'{label}', marker='x', color=colors[i], linestyle=linestyles[i], alpha=alphas[i])

leg = ax.legend(loc="best", shadow=True, fancybox=True, fontsize='x-small')
ax.set(xlabel=f'{x_label}', ylabel=f'{dataset} recall@1',
       title=f'Eval ooDML [{labelspace}] [384]')
ax.set_xticks(x_ticks)
plt.axvline(x=x_ticks[default_split[dataset]]) # add default split indicator (vertical line)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax.grid()

# save plot
customstring = f'msloss'
path_save = f'/export/home/tmilbich/PycharmProjects/dml_pl/experiments/plots/ooDML/eval_{labelspace}_{dataset}'
path_save = "_".join([path_save, customstring])
fig.savefig(path_save + '.png', dpi=600) # .svg'

plt.show()


