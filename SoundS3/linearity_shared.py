from os import path

USING_METRIC = 'R2'
METRIC_DISPLAY = dict(
    ylabel='$R^2$',
    rotation=0,
    labelpad=15,
)

# USING_METRIC = 'diffStd'
# METRIC_DISPLAY = dict(
#     ylabel='Std of Diff', 
# )

# USING_METRIC = 'linearProjectionMSE'
# METRIC_DISPLAY = dict(
#     ylabel='Linear Projection MSE',
# )

# USING_METRIC = 'linearProjectionStdErr'
# METRIC_DISPLAY = dict(
#     ylabel='Linear Projection Std Error', 
# )

SPICE = 'SPICE'

# EXP_GROUPS = [
#     # display name, path name
#     ('VAE aug $\\times 4$, lock $z_\\mathrm{timbre}$', 'vae_symm_4_repeat'), 
#     ('VAE aug $\\times 4$, lock $z_\\mathrm{timbre}$ 10D', 'vae_symm_4_repeat_timbre10d'), 
#     (' AE aug $\\times 4$, lock $z_\\mathrm{timbre}$', 'ae_symm_4_repeat'), 
#     ('VAE aug $\\times 0$, lock $z_\\mathrm{timbre}$', 'vae_symm_0_repeat'), 
#     ('VAE aug $\\times 4$, RNN  $z_\\mathrm{timbre}$', 'vae_symm_4_no_repeat'), 
#     ('SPICE', SPICE), 
# ]

EXP_GROUPS = [
    # display name, path name
    # ('SPS (Ours)', 'nottingham_eighth_accOnly_5000_2_512_easier_gru___checkpoint_9000'), 
    # ('Ours w/o Symmetry', 'nottingham_eighth_accOnly_5000_2_512_easier_gru_noSymm_checkpoint_9000'), 
    # ('$\\beta$-VAE (Baseline)', 'nottingham_eighth_accOnly_5000_2_512_easier_gru_beta_new3_checkpoint_9000'), 
    ('SPS\\textsubscript{VAE}  \linebreak $K$=4 (Ours)', 'scale_singleInst_1dim_0'),
    # ('SPS-VAE1', 'scale_singleInst_1dim_1'),
    # ('SPS-VAE2', 'scale_singleInst_1dim_2'),
    # ('SPS-VAE3', 'scale_singleInst_1dim_3'),
    # ('SPS-VAE4', 'scale_singleInst_1dim_4'),
    # ('SPS-VAE5', 'scale_singleInst_1dim_5'),
    # ('SPS-VAE6', 'scale_singleInst_1dim_6'),
    # ('SPS-VAE7', 'scale_singleInst_1dim_7'),
    # ('SPS-VAE8', 'scale_singleInst_1dim_8'),
    # ('SPS-VAE9', 'scale_singleInst_1dim_9'),
    # ('SPS-AE', 'scale'),
    ('\,SPS\\textsubscript{AE}  \linebreak $K$=4 (Ours)', 'scale_singleInst_1dim_ae_0'),
    # ('SPS-AE1', 'scale_singleInst_1dim_ae_1'),
    # ('SPS-AE2', 'scale_singleInst_1dim_ae_2'),
    # ('SPS-AE3', 'scale_singleInst_1dim_ae_3'),
    # ('SPS-AE4', 'scale_singleInst_1dim_ae_4'),
    # ('SPS-AE5', 'scale_singleInst_1dim_ae_5'),
    # ('SPS-AE6', 'scale_singleInst_1dim_ae_6'),
    # ('SPS-AE7', 'scale_singleInst_1dim_ae_7'),
    # ('SPS-AE8', 'scale_singleInst_1dim_ae_8'),
    # ('SPS-AE9', 'scale_singleInst_1dim_ae_9'),
    # ('SPS-noSymm', 'scale_noSymm'),
    ('\,\,\,\,SPS\\textsubscript{VAE}\linebreak $K$=0 (Ablation)', 'scale_singleInst_1dim_noSymm_0'),
    # ('SPS-noSymm1', 'scale_singleInst_1dim_noSymm_1'),
    # ('SPS-noSymm2', 'scale_singleInst_1dim_noSymm_2'),
    # ('SPS-noSymm3', 'scale_singleInst_1dim_noSymm_3'),
    # ('SPS-noSymm4', 'scale_singleInst_1dim_noSymm_4'),
    # ('SPS-noSymm5', 'scale_singleInst_1dim_noSymm_5'),
    # ('SPS-noSymm6', 'scale_singleInst_1dim_noSymm_6'),
    # ('SPS-noSymm7', 'scale_singleInst_1dim_noSymm_7'),
    # ('SPS-noSymm8', 'scale_singleInst_1dim_noSymm_8'),
    # ('SPS-noSymm9', 'scale_singleInst_1dim_noSymm_9'),
    # ('$\\beta$-VAE (Baseline)', 'scale_betaVAE'),
    ('\,\,\,\,\,SPS\\textsubscript{AE}\linebreak $K$=0 (Ablation)', 'scale_singleInst_1dim_ae_noSymm_0'),
    ('$\\beta$-VAE', 'scale_singleInst_1dim_betaVae_0'),
    # ('betaVAE1', 'scale_singleInst_1dim_betaVae_1'),
    # ('betaVAE2', 'scale_singleInst_1dim_betaVae_2'),
    # ('betaVAE3', 'scale_singleInst_1dim_betaVae_3'),
    # ('betaVAE4', 'scale_singleInst_1dim_betaVae_4'),
    # ('betaVAE5', 'scale_singleInst_1dim_betaVae_5'),
    # ('betaVAE6', 'scale_singleInst_1dim_betaVae_6'),
    # ('betaVAE7', 'scale_singleInst_1dim_betaVae_7'),
    # ('betaVAE8', 'scale_singleInst_1dim_betaVae_8'),
    # ('betaVAE9', 'scale_singleInst_1dim_betaVae_9'),
    # ('$\\beta$-VAE (Baseline)', 'scale_no_Symm_1dim'),
    # ('SPS-AE-noSymm1', 'scale_singleInst_1dim_ae_noSymm_1'),
    # ('SPS-AE-noSymm2', 'scale_singleInst_1dim_ae_noSymm_2'),
    # ('SPS-AE-noSymm3', 'scale_singleInst_1dim_ae_noSymm_3'),
    # ('SPS-AE-noSymm4', 'scale_singleInst_1dim_ae_noSymm_4'),
    # ('SPS-AE-noSymm5', 'scale_singleInst_1dim_ae_noSymm_5'),
    # ('SPS-AE-noSymm6', 'scale_singleInst_1dim_ae_noSymm_6'),
    # ('SPS-AE-noSymm7', 'scale_singleInst_1dim_ae_noSymm_7'),
    # ('SPS-AE-noSymm8', 'scale_singleInst_1dim_ae_noSymm_8'),
    # ('SPS-AE-noSymm9', 'scale_singleInst_1dim_ae_noSymm_9'),
    ('SPICE', SPICE), 
]

TASKS = [
    # path name, display name, x, y, plot style
    (
        'encode', 'Embedding', 
        ('pitch', 'Pitch'), 
        ('z_pitch', '$\\textbf{z}$'),
        dict(
            linestyle='none', 
            marker='.', 
            markersize=3, 
        ), 
    ), 
    (
        'decode', 'Synthesis', 
        # ('z_pitch', '$z_\\mathrm{pitch}$'),
        ('z_pitch', '$\\textbf{z}$'),
        ('yin_pitch', 'Detected Pitch'),
        dict(
            linestyle='none', 
            marker='.', 
            markersize=3, 
        ), 
    ), 
]

DATA_SETS = [
    # path name, display name
    # ('train_set', 'Train'), 
    ('test_set', 'Test'), 
]

RESULT_PATH = './linearityEvalResults/%s_%s_%s/'
SPICE_PATH = './SPICE_results/result_short.txt'

COMMON_INSTRUMENTS = ['Accordion',
    # 'Piano', 'Accordion', 'Clarinet', 'Electric Piano', 
    # 'Flute', 'Guitar', 'Saxophone', 'Trumpet', 'Violin', 
    # 'Church Bells', 
]

def readXYFromDisk(
    is_SPICE, 
    result_path, 
    x_path, y_path, 
):
    data = {}
    if is_SPICE:
        if 'decode' in result_path or 'train_set' in result_path:
            raise NoSuchSpice
        with open(SPICE_PATH, 'r') as f:
            for line in f:
                line: str = line.strip()
                line = line.split('cleanTrain_bend/')[1]
                filename, z_pitch = line.split('.wav ')
                z_pitch = float(z_pitch)
                instrument_name, pitch = filename.split('-')
                pitch = float(pitch)
                if instrument_name in COMMON_INSTRUMENTS:
                    if instrument_name not in data:
                        data[instrument_name] = ([], [])
                    X, Y = data[instrument_name]
                    X.append(pitch)
                    Y.append(z_pitch)
    else:
        for instrument_name in COMMON_INSTRUMENTS:
            if instrument_name not in COMMON_INSTRUMENTS:
                continue
            X = []
            Y = []
            def f(output: list, s: str):
                with open(path.join(
                    result_path.replace('test_set_',''), instrument_name + f'_{s}.txt'
                ), 'r') as f:
                    for line in f:
                        output.append(float(line.strip()))
            f(X, x_path)
            f(Y, y_path)
            data[instrument_name] = (X, Y)
    return data

class NoSuchSpice(Exception): pass
