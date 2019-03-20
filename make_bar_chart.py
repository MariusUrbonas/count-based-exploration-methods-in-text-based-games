import glob
import numpy as np
import os
import pickle
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import sys


NUM_EPOCHS = 400
NUM_QUEST_LENGTHS = 4
WIDTH = 0.8
STATS_FOLDERS = {  # 'folder': ('label', '#hexcolor')
    'banana_baseline': ('DQN (Baseline)', '#1b9e77'),
    'dqn-ubc-sa': ('DQN-UCB-SA', '#d95f02'),
    'dqn-mbie-eb': ('DQN-MBIE-EB', '#7570b3'),
    'banana_cumulative': ('DQN-S+', '#e7298a'),
    'banana_episodic': ('DQN-S++', '#a6761d'),
    'cool_agent': ('DQN-KM++', '#e6ab02')
}

# Get parameters
try:
    output_name = sys.argv[1]
except:
    print('Usage: pythonw make_graphs.py output_name')
    exit()

# Create trace for each model
traces = []
for index, stats_folder in enumerate(STATS_FOLDERS):
    
    # Get data for each quest length
    data_lists = []
    for quest_length in range(1, NUM_QUEST_LENGTHS + 1):
        data_lists.append([])
        stats_files = glob.glob('experiments/{}/stats/*ql-{}*.pickle'.format(stats_folder, quest_length))
        for stats_file in stats_files:
            with open(stats_file, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                if 'obs_set' in data:
                    del(data['obs_set'])
                data_lists[-1].append([data[epoch]['scores'] for epoch in data][:NUM_EPOCHS])
    
    data_np = np.array(data_lists)
    label, color = STATS_FOLDERS[stats_folder]

    # Mean of all epochs
    traces.append(go.Bar(
        x=['Quest Length {}'.format(ql + 1) for ql in range(NUM_QUEST_LENGTHS)],
        y=np.mean(data_np, (1, 2, 3)),
        name=label,
        marker=dict(
            color=color
        ),
        error_y=dict(
            type='data',
            array=np.std(data_np, (1, 3)).mean((1,)),
            visible=True,
            thickness=0.3,
            width=2
        )
    ))

    # Mean of last 50 epochs
    traces.append(go.Bar(
        x=['Quest Length {}'.format(ql + 1) for ql in range(NUM_QUEST_LENGTHS)],
        y=np.mean(data_np[:, :, -50:, :], (1, 2, 3)),
        marker=dict(
            color=color,
            opacity=0.3
        ),
        error_y=dict(
            type='data',
            array=np.std(data_np[:, :, -50:, :], (1, 3)).mean((1,)),
            visible=True,
            thickness=0.3,
            width=2
        ),
        showlegend=False,
    ))

layout = go.Layout(
    barmode='group',
    bargroupgap=0.15,
    legend=dict(
        orientation='h',
        font=dict(
            family='sans-serif',
            size=8
        ),
    ),
    xaxis=dict(
        tickfont=dict(
            family='sans-serif',
            size=10
        ),
    ),
    yaxis=dict(
        title='Mean Score',
        titlefont=dict(
            family='sans-serif',
            size=10
        ),
        tickfont=dict(
            family='sans-serif',
            size=10
        ),
    )
)
fig = go.Figure(data=traces, layout=layout)
pio.write_image(fig, 'figures/{}.pdf'.format(output_name))
