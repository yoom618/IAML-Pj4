import numpy as np
import os
import pretty_midi


def get_pianoroll_list(data_path):
    '''
    :param data_path: data directory
    :return: list of piano rolls
    '''
    piano_rolls = []
    for path, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.mid'):
                piano_roll = midi_to_pianoroll(os.path.join(path, file))
                piano_rolls.append(piano_roll / 90)

    return np.array(piano_rolls)


def midi_to_pianoroll(path):
    '''
    :param path: midi path
    :return: piano roll of shape [256, 128]
    '''
    pm = pretty_midi.PrettyMIDI(path)
    pianoroll = pm.get_piano_roll(fs=8)
    pianoroll = np.array(pianoroll, dtype=np.int32).T
    if pianoroll.shape[0] > 256:
        return pianoroll[:256, :]
    elif pianoroll.shape[0] < 256:
        mask = 256 - pianoroll.shape[0]
        return np.pad(pianoroll, [(0, mask), (0, 0)], 'constant')
    else:
        return pianoroll


def pianoroll_to_midi(pianoroll, filename):
    '''
    :param pianoroll: piano roll of shape [256, 128]
    :param filename: save path
    '''
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    frames, notes = pianoroll.shape

    for pitch in range(notes):
        velocity_changes = np.nonzero(np.diff(pianoroll[:, pitch]))[0]
        if velocity_changes.size > 0:
            prev_velocity = pianoroll[velocity_changes[0], pitch]
            prev_time = 0.0
            for i, time in enumerate(velocity_changes):
                if prev_velocity != 0:
                    pm_note = pretty_midi.Note(
                        velocity=prev_velocity,
                        pitch=pitch,
                        start=prev_time,
                        end=(time + 1) / 8)
                    instrument.notes.append(pm_note)
                    prev_velocity = 0
                else:
                    prev_time = (time + 1) / 8
                    prev_velocity = 90

    pm.instruments.append(instrument)
    pm.write(filename)
