import numpy as np


def extract_all_features_method1_a(pm_list, all_lyrics_list, seq_length, number_of_all_seq):
    """
    This function creates features from the melody of each song. The features' values are relative to each word
    according to her index in the song.
    The features are based on the piano roll of the pretty midi object which is a matrix of size
    (128, amount of time slices) that represent the sum of velocities of each pitch over time.
    The piano roll in sliced according to the average time per word in the song.
    :param pm_list: a list of all pretty midi objects
    :param all_lyrics_list: a list of strings where each string is the lyrics of one song
    :param seq_length: the length of one sequence
    :param number_of_all_seq: the total amount of sequences to create
    :return: an array of features' values for each sequence
    """
    i = 0
    k = 0
    final_seq_features = np.zeros((number_of_all_seq, seq_length, 3))

    for pm, lyrics in zip(pm_list, all_lyrics_list):
        print(f'song {i}')
        song_lyrics_split = lyrics.split(" ")
        total_words_in_song = len(song_lyrics_split)
        pm.remove_invalid_notes()
        first_note_in_song_time = find_first_note(pm)
        end_time_pm = pm.get_end_time()
        total_song_time = end_time_pm - first_note_in_song_time
        avg_time_for_word = total_song_time/total_words_in_song

        start_times = [(idx * avg_time_for_word + first_note_in_song_time) for idx in range(total_words_in_song)]
        end_times = [min(start_time + avg_time_for_word, end_time_pm) for start_time in start_times]
        fs = 200
        pr_times = np.arange(first_note_in_song_time, end_time_pm, 1. / fs)
        piano_roll = pm.get_piano_roll(times=pr_times)
        samples_per_word = int(np.floor(fs * avg_time_for_word))
        samples_starts = [idx*samples_per_word for idx in range(total_words_in_song)]
        samples_ends = [samples_start_i + samples_per_word for samples_start_i in samples_starts]
        piano_rolls = [piano_roll[:, start_i:end_i] for start_i, end_i in zip(samples_starts, samples_ends)]
        most_common_pitches = np.array(list(map(extract_most_common_pitch, piano_rolls)))
        average_velocity = np.array(list(map(extract_average_velocity, piano_rolls)))
        pms = [pm for i in range(len(start_times))]
        notes_density = np.array(list(map(extract_note_density, pms, start_times, end_times)))
        all_features = (np.vstack((most_common_pitches, average_velocity, notes_density))).T
        number_of_seq = total_words_in_song - seq_length
        for j in range(number_of_seq):
            seq_feature = all_features[j: j + seq_length, :]
            final_seq_features[k, :, :] = seq_feature
            k += 1
        i += 1

    # normalize
    first_max = np.max(final_seq_features[:, :, 1])
    first_min = np.min(final_seq_features[:, :, 1])
    final_seq_features[:, :, 1] = (final_seq_features[:, :, 1] - first_min) / (first_max - first_min)
    second_max = np.max(final_seq_features[:, :, 2])
    second_min = np.min(final_seq_features[:, :, 2])
    final_seq_features[:, :, 2] = (final_seq_features[:, :, 2] - second_min) / (second_max - second_min)

    print(final_seq_features.shape)
    return final_seq_features


def extract_all_features_method1_b(pm_list, all_lyrics_list, seq_length, number_of_all_seq):
    """
    This function creates features from the melody of each song. The features' values are relative to he entire song.
    The features are based on the piano roll of the pretty midi object which is a matrix of size
    (128, amount of time slices) that represent the sum of velocities of each pitch over time.
    :param pm_list: a list of all pretty midi objects
    :param all_lyrics_list: a list of strings where each string is the lyrics of one song
    :param seq_length: the length of one sequence
    :param number_of_all_seq: the total amount of sequences to create
    :return: an array of features' values for each sequence
    """
    i = 0
    j = 0
    final_seq_features = np.zeros((number_of_all_seq, seq_length, 3))

    for pm, lyrics in zip(pm_list, all_lyrics_list):
        print(f'song {j}')
        song_lyrics_split = lyrics.split(" ")
        total_words_in_song = len(song_lyrics_split)
        pm.remove_invalid_notes()
        fs = 200
        piano_roll = pm.get_piano_roll(fs)
        most_common_pitch = extract_most_common_pitch(piano_roll)
        average_velocity = extract_average_velocity(piano_roll)
        notes_density = extract_note_density_for_pm(pm)
        all_features = np.array([most_common_pitch, average_velocity, notes_density])
        number_of_seq = total_words_in_song - seq_length

        final_seq_features[i:i + number_of_seq, :, :] = all_features
        i += number_of_seq
        j += 1

    # normalize
    first_max = np.max(final_seq_features[:, :, 1])
    first_min = np.min(final_seq_features[:, :, 1])
    second_max = np.max(final_seq_features[:, :, 2])
    second_min = np.min(final_seq_features[:, :, 2])
    final_seq_features[:, :, 1] = (final_seq_features[:, :, 1] - first_min) / (first_max - first_min)
    final_seq_features[:, :, 2] = (final_seq_features[:, :, 2] - second_min) / (second_max - second_min)

    print(final_seq_features.shape)
    return final_seq_features


def extract_most_common_pitch(piano_roll):
    """
    This function returns the pitch (accord) whose velocity was the highest, meaning it was the most heard
    in this time frame
    :param piano_roll: piano roll matrix
    :return: a number from 0 to 1 127 corresponding to the pitch.
             Originally 0-127 but normalized by dividing in 127
    """
    sum_of_rows = np.sum(piano_roll, axis=1)
    max_pitch = np.argmax(sum_of_rows, axis=0)
    return max_pitch/127


def extract_note_density(pm, start_time, end_time):
    """
    This function returns the amount of notes in the given time frame
    :param pm: pretty midi object
    :param start_time:
    :param end_time:
    :return: amount of notes in the time frame
    """
    all_notes = [note_i for note in [inst.notes for inst in pm.instruments] for note_i in note
                 if ((end_time >= note_i.start >= start_time) or (end_time >= note_i.end >= start_time))]
    notes_counter = len(all_notes)
    return notes_counter


def extract_note_density_for_pm(pm):
    """
    This function returns the amount of notes in the given time frame
    :param pm: pretty midi object
    :return: amount of notes in the time frame
    """
    all_notes = [note_i for note in [inst.notes for inst in pm.instruments] for note_i in note]
    notes_counter = len(all_notes)
    return notes_counter


def extract_average_velocity(piano_roll):
    """
    This function returns average velocity if the given piano roll
    :param piano_roll: piano roll matrix
    :return: average velocity
    """
    avg_velocity = np.average(piano_roll)
    return avg_velocity


def find_first_note(pm):
    """
    This function returns the start time of the first note in the melody
    :param pm: pretty midi object
    :return: (float) the start time of the first note in the melody
    """
    all_notes = [note_i for note in [inst.notes for inst in pm.instruments] for note_i in note]
    first_note_time = all_notes[0].start
    return first_note_time


def extract_all_features_method2(pm_list, lyrics_list, seq_length, number_of_all_seq):
    """
    This function creates for each song a vector of size 128. The values are 1 in the indexes corresponding to
    instruments ids if this instrument appeared in the pm object of the song.
    It return an array of size (number of all sequences, sequence length, 128)
    :param pm_list: a list of all pretty midi objects
    :param lyrics_list: a list of strings where each string is the lyrics of one song
    :param seq_length: the length of one sequence
    :param number_of_all_seq: the total amount of sequences to create
    :return: an array of features' values for each sequence
    """
    features = np.zeros((len(pm_list), 128))
    for pm_idx, pm in enumerate(pm_list):
        pm.remove_invalid_notes()
        instruments_ids = [inst.program for inst in pm.instruments]
        features[pm_idx, instruments_ids] = 1

    final_features_seq = np.zeros((number_of_all_seq, seq_length, 128))

    i = 0
    for lyrics_idx, lyrics in enumerate(lyrics_list):
        song_lyrics_split = lyrics.split(" ")
        total_seq_in_song = len(song_lyrics_split) - seq_length
        final_features_seq[i:i+total_seq_in_song, :, :] = features[lyrics_idx, :]
        i += total_seq_in_song

    return final_features_seq
