import os

ROOT_REPO_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
WORD_AND_SPEAKER_ENCODINGS_PATH = os.path.join(ROOT_REPO_DIR, 'robustness', 'audio_functions', 'word_and_speaker_encodings_jsinv3.pckl')
ASSETS_PATH = '/om2/user/jfeather/projects/component_synthesis/model_metamers_pytorch/assets/'#os.path.join(ROOT_REPO_DIR, 'assets')
WORDNET_ID_TO_HUMAN_PATH = os.path.join(ROOT_REPO_DIR, 'analysis_scripts', 'wordnetID_to_human_identifier.txt')
# IMAGENET_PATH = None # /om/data/public/imagenet/images_complete/
IMAGENET_PATH = '/om/data/public/imagenet/images_complete/ilsvrc/'
# JSIN_PATH = None
JSIN_PATH = '/om4/group/mcdermott/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets'

# fMRI dataset paths
fMRI_DATA_PATH = os.path.join(ASSETS_PATH, 'fMRI_natsound_data')
