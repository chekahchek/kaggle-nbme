FC_DROPOUT = 0.2
SEED = 42
NUM_WORKERS = 4
N_FOLD = 5
TRN_FOLD = [0, 1, 2, 3, 4]
BEST_TH = 0.65
DEBUG = False
MODELS = {
    'google-electra-large-discriminator' : 0.33,
    'microsoft-deberta-v3-large-pl' : 0.46,
    'microsoft-deberta-v3-large-retrained' : 0.44,
    'public-deberta-large' : 0.22,
    'microsoft-deberta-v2-xlarge': 0.32
}

MODELS_LEN = {
    'google-electra-large-discriminator' : 344,
    'microsoft-deberta-v3-large-pl' : 354,
    'microsoft-deberta-v3-large-retrained' : 354,
    'public-deberta-large' : 466,
    'microsoft-deberta-v2-xlarge': 351
}    

MODELS_BATCH_SIZE = {
    'google-electra-large-discriminator' : 128,
    'microsoft-deberta-v3-large-pl' : 64,
    'microsoft-deberta-v3-large-retrained' : 64,
    'public-deberta-large' : 64,
    'microsoft-deberta-v2-xlarge' : 64
} 

MODELS_PATH = {
    'google-electra-large-discriminator' : 'electralarge',
    'microsoft-deberta-v3-large-pl' : 'debertalargepl',
    'microsoft-deberta-v3-large-retrained' : 'debertav3largeretrained',
    'public-deberta-large' : 'debertalarge',
    'microsoft-deberta-v2-xlarge' : 'debertav2xlarge'
}
