import os
from os import path
import shutil
import pickle
import random

import pretty_midi as pm
from music21.instrument import Instrument, Piano
try:
    # my fork
    from midi2audio_fork.midi2audio import FluidSynth
except ImportError:
    # fallback
    from midi2audio import FluidSynth
import librosa
import pretty_midi
import soundfile
import numpy as np
from tqdm import tqdm

from dataset_config import *
from intruments_and_ranges import intruments_ranges

# MODIFIED GLOBALS
N_STEPS = 64
DATASET_PATH = './datasets_out/nottingham_eights_pool_accOnly_5000_easier' # WAV output
DATASET_IN_PATH = './datasets_in/nottingham-dataset-master/MIDI/melody'

# GLOBALS
PROLONG = False
BEND_RATIO = .5  # When MIDI specification is incomplete...
BEND_MAX = 8191
GRACE_END = .1
TEMP_MIDI_FILE = path.abspath('./temp/temp.mid')
TEMP_WAV_FILE  = path.abspath('./temp/temp.wav')
SOUND_FONT_PATH = './FluidR3_GM/FluidR3_GM.sf2'
# DATASET_PATH = './datasets/cleanTrain'
# DATASET_PATH = './datasets/single_note'

# SOUND_FONT_PATH = './FluidR3_GM/GeneralUser GS v1.471.sf2'
# DATASET_PATH = './datasets/cleanTrain_GU'

if PROLONG:
    DATASET_PATH += '_long'
    assert N_HOPS_PER_NOTE == 4
    N_HOPS_PER_NOTE = 30

    ENCODE_STEP = N_HOPS_PER_NOTE + N_HOPS_BETWEEN_NOTES
    N_SAMPLES_PER_NOTE = HOP_LEN * N_HOPS_PER_NOTE
    N_SAMPLES_BETWEEN_NOTES = HOP_LEN * N_HOPS_BETWEEN_NOTES
    NOTE_DURATION = N_SAMPLES_PER_NOTE / SR
    NOTE_INTERVAL = (N_SAMPLES_PER_NOTE + N_SAMPLES_BETWEEN_NOTES) / SR

SONG_LEN = (
    N_SAMPLES_PER_NOTE + N_SAMPLES_BETWEEN_NOTES
) * N_STEPS

fs = FluidSynth(SOUND_FONT_PATH, sample_rate=SR)

def notes_to_d_pitches(notes, n_bins, step_size, easy):
    d_pitches = ['rest' for _ in range(n_bins)]
    for idx, val in enumerate(d_pitches):
        if val == 'sustain':
            continue
        for note in notes:
            if note.start <= idx * step_size and note.end > idx * step_size:
                d_pitches[idx] = note.pitch
                for i in range(idx + 1, int(min(idx + max((note.end - note.start-1e-6) // step_size + 1, 0), n_bins))):
                    if easy:
                        d_pitches[i] = note.pitch
                    else:
                        d_pitches[i] = 'sustain'
                break
    if easy:
        # Remove rests
        for idx, val in enumerate(d_pitches):
            if val == 'rest' and idx != 0:
                d_pitches[idx] = d_pitches[idx-1]
        
    return d_pitches

def slice_notes(notes, n_slice=1000, slice_size=2):
    out = [[] for _ in range(n_slice)]
    
    # Cut notes crossing barlines and bin into slices 
    for note in notes:
        if note.end >= n_slice * slice_size:
            continue
        # Skip long sustains
        if note.end - note.start > slice_size:
            continue

        if note.start // slice_size != note.end // slice_size and not (note.end % slice_size == 0):
            out[int(note.start // slice_size)].append(pretty_midi.Note(start = note.start % slice_size, end = slice_size, pitch = note.pitch, velocity = 100))
            out[int(note.end // slice_size)].append(pretty_midi.Note(start = 0, end = note.end % slice_size, pitch = note.pitch, velocity = 100))
        else:
            out[int(note.start // slice_size)].append(pretty_midi.Note(start = note.start % slice_size, end = note.end % slice_size, pitch = note.pitch, velocity = 100))
    return out

def remove_repetition(d_pitch, n):
    for idx in range(n, len(d_pitch)):
        if len(set(d_pitch[idx - n: idx + 1])) == 1:
            d_pitch[idx] = random.choice(d_pitch[: idx])
            while len(set(d_pitch[idx - n: idx + 1])) == 1:
                d_pitch[idx] = random.choice([i for i in range(48, 72)])

    # Remove hanging sustains
    for idx in range(1, len(d_pitch)):
        if d_pitch[idx] == 'sustain' and d_pitch[idx-1] == 'rest':
                d_pitch[idx] = random.choice([i for i in range(48, 72)])
    return d_pitch

def make_natural_mel_dataset(size=1e10, n_slice=4, slice_size=2, n_bins=16, step_size=60/80/4, block_ngram=2, accordion_only=False, easy=False):
    # Initialise 
    try:
        shutil.rmtree(DATASET_PATH)
    except FileNotFoundError:
        pass
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Preproc all melodies 
    d_pitches = []
    max_p = 0
    min_p = 100
    for file in tqdm(os.listdir(DATASET_IN_PATH)):
        midi_path = f'{DATASET_IN_PATH}/{file}'
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = midi.instruments[0].notes
    
        notes_slices = slice_notes(notes, n_slice, slice_size)
        
        for idx, notes_slice in enumerate(notes_slices):
            if notes_slice != []:
                for note in notes_slice:
                    if note.pitch > max_p:
                        max_p = note.pitch
                    if note.pitch < min_p:
                        min_p = note.pitch
                d_pitch = notes_to_d_pitches(notes_slice, n_bins, step_size, easy=easy)

                if block_ngram > 0:
                    d_pitch = remove_repetition(d_pitch, block_ngram)
                d_pitches.append([d_pitch, file, idx])
                if len(d_pitches) > size:
                    break
        if len(d_pitches) > size:
                break
    print(f'#Melody slices: {len(d_pitches)}')
    print(max_p, min_p)
    exit()
    # Synthesize for each instrument
    index = []
    for instrument, pitch_range in tqdm(intruments_ranges):
        pitches_audio = {}
        for pitch in pitch_range:
            audio = synthOneNote(
                fs, pitch, instrument, 
            )
            pitches_audio[pitch] = audio
            dtype = audio.dtype
            audio[-FADE_OUT_N_SAMPLES:] = audio[
                -FADE_OUT_N_SAMPLES:
            ] * FADE_OUT_FILTER

        # Loop over all melodies
        for n_mel, [d_pitch, folder, idx] in enumerate(d_pitches):

            # skip if any pitch is out of range for the instrument
            out_of_range = False
            for d in d_pitch:
                if d not in ['rest', 'sustain'] and d not in pitch_range:
                    out_of_range = True
            if out_of_range:
                continue
                
            # Generate the wav file 
            song = GenSong(pitches_audio, d_pitch, dtype)

            index.append((instrument.instrumentName, folder, idx))
            soundfile.write(path.join(
                DATASET_PATH, 
                f'{instrument.instrumentName}-{folder}-{idx}.wav', 
            ), song, SR)
    
        if accordion_only:
            break
    
    with open(path.join(DATASET_PATH, 'index.pickle'), 'wb') as f:
        pickle.dump(index, f)



def synthOneNote(
    fs: FluidSynth, pitch: float, instrument: Instrument, 
    temp_wav_file=TEMP_WAV_FILE, verbose=False, 
):
    # make midi
    music = pm.PrettyMIDI()
    ins = pm.Instrument(program=instrument.midiProgram)
    rounded_pitch = int(round(pitch))
    note = pm.Note(
        velocity=100, pitch=rounded_pitch, 
        start=0, end=NOTE_DURATION*30 + GRACE_END, 
    )
    pitchBend = pm.PitchBend(
        round((pitch - rounded_pitch) * BEND_MAX * BEND_RATIO), 
        time=0, 
    )
    if verbose:
        print(rounded_pitch, ',', pitchBend.pitch)
    ins.notes.append(note)
    ins.pitch_bends.append(pitchBend)
    music.instruments.append(ins)
    music.write(TEMP_MIDI_FILE)
    fs.midi_to_audio(os.path.join(TEMP_MIDI_FILE), os.path.join(temp_wav_file), verbose=False)

    # read wav
    audio, sr = librosa.load(temp_wav_file, SR)
    assert sr == SR
    return audio

def GenSong(pitches_audio, d_pitches, dtype):

    song = np.zeros((SONG_LEN, ), dtype=dtype)
    cursor = N_SAMPLES_BETWEEN_NOTES
    for idx, d_pitch in enumerate(d_pitches):
        if d_pitch == 'sustain':
            continue

        n_sustain = 1
        if d_pitch == 'rest':
            audio = [0 for _ in range(N_SAMPLES_PER_NOTE)]

        else:
            # find sustain length
            for pitches in d_pitches[idx+1:]:
                if pitches == 'sustain':
                    n_sustain += 1
                else:
                    break
            pitch = d_pitch
            try:
                audio = pitches_audio[pitch][:N_SAMPLES_PER_NOTE * n_sustain + N_SAMPLES_BETWEEN_NOTES * (n_sustain - 1)]
            except KeyError:
                return
        song[
            cursor : cursor + N_SAMPLES_PER_NOTE * n_sustain + N_SAMPLES_BETWEEN_NOTES * (n_sustain - 1)
        ] = audio
        cursor += N_SAMPLES_PER_NOTE * n_sustain
        cursor += N_SAMPLES_BETWEEN_NOTES * n_sustain
    assert cursor - N_SAMPLES_BETWEEN_NOTES == SONG_LEN
    return song

make_natural_mel_dataset(size=400000, n_slice=10000, slice_size=16, n_bins=N_STEPS, step_size=1/4, accordion_only=True, easy=True, block_ngram=2)
