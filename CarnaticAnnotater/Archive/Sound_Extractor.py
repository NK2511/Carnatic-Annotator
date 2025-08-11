import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris
from scipy.io.wavfile import read
import math
from pydub import AudioSegment

# Add the path to the directory containing the modules
dft_model_path = r'C:\Users\nandh\sms-tools-master\software\models'
util_functions_path = r'C:\Users\nandh\sms-tools-master\software\models'
sine_model_path = r'C:\Users\nandh\sms-tools-master\software\models'

sys.path.append(dft_model_path)
sys.path.append(util_functions_path)
sys.path.append(sine_model_path)

import dftModel as DFT
import utilFunctions as UF
import sineModel as SM


def convert_to_mono_wav(input_file, output_file):
    """Convert MP3 or stereo file to mono WAV."""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(output_file, format="wav")


def f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et):
    """Fundamental frequency detection of a sound using twm algorithm."""
    if (minf0 < 0):
        raise ValueError("Minimum fundamental frequency (minf0) smaller than 0")
    if (maxf0 >= 10000):
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")
    if (H <= 0):
        raise ValueError("Hop size (H) smaller or equal to 0")

    hN = N // 2
    hM1 = int(math.floor((w.size + 1) / 2))
    hM2 = int(math.floor(w.size / 2))
    x = np.append(np.zeros(hM2), x)
    x = np.append(x, np.zeros(hM1))
    pin = hM1
    pend = x.size - hM1
    fftbuffer = np.zeros(N)
    w = w / sum(w)
    f0 = []
    f0stable = 0

    while pin < pend:
        x1 = x[pin - hM1:pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
        ipfreq = fs * iploc / N
        f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)
        if ((f0stable == 0) & (f0t > 0)) or ((f0stable > 0) & (np.abs(f0stable - f0t) < f0stable / 5.0)):
            f0stable = f0t
        else:
            f0stable = 0
        f0 = np.append(f0, f0t)
        pin += H

    return f0


# Input file
input_file = r"C:\Users\nandh\Downloads\Amaran Violin.mp3"
output_file = r"C:\Users\nandh\Downloads\Amaran_Violin_mono.wav"

# Convert to mono WAV
convert_to_mono_wav(input_file, output_file)

# Parameters
fs, x = read(output_file)  # Read audio file
x = x / np.max(np.abs(x))  # Normalize audio
w = blackmanharris(1024)  # Analysis window
N = 2048  # FFT size
H = 512  # Hop size
t = -80  # Threshold
minf0 = 300  # Minimum f0 in Hz
maxf0 = 1000  # Maximum f0 in Hz
f0et = 5  # Error threshold

# Compute f0
f0 = f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)

# Plot f0
time = np.arange(len(f0)) * H / fs  # Time axis
plt.figure(figsize=(10, 6))
plt.plot(time, f0, label='f0 (Hz)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency (f0)')
plt.grid()
plt.legend()
plt.show()

f0_file_path = 'initial_f0_values_temp.txt'
with open(f0_file_path, 'w') as f:
    for freq in f0:
        f.write(f"{freq}\n")