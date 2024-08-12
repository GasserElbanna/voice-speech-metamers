import scipy
import numpy as np
import glob

paths = '/om2/user/salavill/misc/voice-speech-metamers/metamers_pipeline/joint_model/joint_metamer.npy'


scipy.io.wavfile.write(
    glob.glob('*.wav')[0], 16000, np.load(glob.glob('*_2_*.npy')[0])[0]
)