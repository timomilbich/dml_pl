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

dataset_label_lookup = {
    'cub200': 'CUB200-2011',
    'cars196': 'CARS196',
    'sop': 'Online Products (SOP)'
}

# get wandb run data
name_list, group_list, summary_list, config_list, data_list = experiments.get_wandb_data()

project_group = 'ooDML_msloss'
labelspace = 'arch'
dataset = 'sop' # 'cub200'
filter_substrings = ['100epoch', 'gpu2_sop', 'vitS8Dino_bs30_c'] # 'vitS8Dino_bs30_c'

# filter runs data - project group
ids_valid = [i for i, group in enumerate(group_list) if group == project_group]
name_list = [name_list[i] for i in ids_valid]
data_list = [data_list[i] for i in ids_valid]

# filter runs data - dataset
ids_valid = [i for i, d in enumerate(data_list) if d['dataset'] == dataset]
name_list = [name_list[i] for i in ids_valid]
data_list = [data_list[i] for i in ids_valid]

# filter runs data - substrings in name
ids_valid = [i for i, n in enumerate(name_list) if not any(map(n.__contains__, filter_substrings))]
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
n_x_ticks = []
for label in labels_unique:
    recs = recall_seqs[label]
    split_ids = split_seqs[label]

    # sort recall values
    recs = [x for _, x in sorted(zip(split_ids, recs))]

    # reassign
    recall_seqs[label] = recs
    split_seqs[label] = sorted(split_ids)

    n_x_ticks.append(len(recs))
n_x_ticks = min(n_x_ticks)

# cut data to n_x_ticks - to still be able to visualize experiments in progress
for label in labels_unique:
    recall_seqs[label] = recall_seqs[label][:n_x_ticks]
    split_seqs[label] = split_seqs[label][:n_x_ticks]

# plotting
x_label = 'split_id'
x_ticks = list(range(1, n_x_ticks + 1)) # e.g. split ids
colors = ['green', 'red', 'blue', 'magenta', 'yellow', 'cyan', 'navy', 'orange', 'deeppink', 'springgreen', 'pink', 'darkkhaki']
alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# fig, ax = plt.subplots(figsize=(5, 4))
# for i, label in enumerate(labels_unique):
#     linestyle = 'dashed' if any(map(label.__contains__, ['resnet', 'bninception'])) else 'solid'
#     ax.plot(x_ticks, recall_seqs[label], label=f'{label}', marker='x', color=colors[i], linestyle=linestyle, alpha=alphas[i])
#
# leg = ax.legend(loc="best", shadow=True, fancybox=True, fontsize='xx-small')
# ax.set(xlabel=f'{x_label}', ylabel=f'{dataset} recall@1',
#        title=f'Eval ooDML [{labelspace}] [384]')
# ax.set_xticks(x_ticks)
# plt.axvline(x=x_ticks[default_split[dataset]]) # add default split indicator (vertical line)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
# ax.grid()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# absolute performance plot
for i, label in enumerate(labels_unique):
    linestyle = 'dashed' if any(map(label.__contains__, ['resnet', 'bninception'])) else 'solid'
    ax1.plot(x_ticks, recall_seqs[label], label=f'{label}', marker='x', color=colors[i], linestyle=linestyle, alpha=alphas[i])

leg = ax1.legend(loc="best", shadow=True, fancybox=True, fontsize='xx-small')
ax1.set(xlabel=f'{x_label}', ylabel=f'{dataset_label_lookup[dataset]} recall@1',
       title=f'Absolute recall performance [{labelspace}] [384]')
ax1.set_xticks(x_ticks)
plt.axvline(x=x_ticks[default_split[dataset]]) # add default split indicator (vertical line)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax1.grid()

# relative performance plot (reference value: performance for split 1)
for i, label in enumerate(labels_unique):
    linestyle = 'dashed' if any(map(label.__contains__, ['resnet', 'bninception'])) else 'solid'
    ax2.plot(x_ticks, [recall_seqs[label][0] - v for v in recall_seqs[label]], label=f'{label}', marker='x', color=colors[i], linestyle=linestyle, alpha=alphas[i])

leg = ax2.legend(loc="best", shadow=True, fancybox=True, fontsize='xx-small')
ax2.set(xlabel=f'{x_label}', ylabel=f'{dataset_label_lookup[dataset]} recall@1 difference',
       title=f'Relative recall performance w.r.t. split 1 [{labelspace}] [384]')
ax2.set_xticks(x_ticks)
plt.axvline(x=x_ticks[default_split[dataset]]) # add default split indicator (vertical line)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax2.grid()

# save plot
customstring = f'msloss_dino60_abs_rel'
path_save = f'/export/home/tmilbich/PycharmProjects/dml_pl/experiments/plots/ooDML/eval_{labelspace}_{dataset}'
path_save = "_".join([path_save, customstring])
fig.savefig(path_save + '.png', dpi=600) # .svg'

plt.show()


