""" EEG file read functions. Returns pandas dataframe """
# needed to add imports
import pandas as pd
import numpy as np
import gzip
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import pickle
import os
from collections import defaultdict
import tqdm
from scipy import signal


senlist_known = [
    'AF1', 'AF2', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4',
    'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ',
    'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1',
    'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FP1', 'FP2',
    'FPZ', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2',
    'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO1', 'PO2', 'PO7',
    'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8', 'X', 'Y',
    'nd']
match_types = ['obj', 'nomatch', 'match']
SAMP_FREQ = 256.
default_metadata_filen = 'eeg_file_metadata.csv'
files_skip_processing = ['.DS_Store', 'README']
large_dataset_prefix = 'large_data_set'


def pass_through(x, y, xlabel, ylabel, fs=None):
    return x, y, xlabel, ylabel


def PSD_on_row_data(x, y, xlabel, ylabel, fs=None):
    f, Pxx = signal.periodogram(y, fs=(1. if fs is None else fs), axis=0)
    return f[1:], Pxx[1:], 'frequency (Hz)', 'PSD (V^2/Hz)'


def fft_on_row_data(x, y, xlabel, ylabel, fs=None):  # complex2real=np.abs):
    """ get the fft of the data, by default complex2real
        produces the magnitude, but could also use np.real
    """
    fft_coeff = np.fft.rfft(y, axis=0)
    ncoeff = fft_coeff.shape[0]
    if fs is not None:
        f = fs/2 * np.arange(0, ncoeff) / ncoeff
    else:
        f = 0.5 * np.arange(0, ncoeff) / ncoeff
    # return f[1:], complex2real(fft_coeff[1:]), 'frequency (Hz)', 'FFT coeff'
    # outarray = np.zeros(2 * (ncoeff - 1))
    # outarray[:(ncoeff-1)] = np.real(fft_coeff[1:])
    # outarray[(ncoeff-1):] = np.real(fft_coeff[1:])
    return (np.concatenate([-np.flip(f[1:], axis=0), f[1:]]), np.concatenate(
        [np.imag(np.flipud(fft_coeff[1:])), np.real(fft_coeff[1:])]),
        'frequency (Hz) (imag | real)', 'FFT coeff level (uV)')


class GetClassId(dict):
    """ simple class to assign new ID if not available, otherwise return
        previously found ID, behaves like a dictionary
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.next_cls_id = 0

    def __getitem__(self, subject_id):
        if subject_id not in self:
            self[subject_id] = self.next_cls_id
            self.next_cls_id += 1
        return super().__getitem__(subject_id)


def get_blocked_data(files_list=None, use_saved_meta_data=True,
        use_onehot_cat_feat=False,
        multiclass=False, pre_split=True, prefix_directory=None,
        process_data=pass_through, pickle_name=None, debug=1):
    """ get the data in a format convenient to torch-based
        model building and training
    """
    assert not((files_list is None) and (pickle_name is None)), (
        "both files_list and pickle_name cannot be undefined")

    if pickle_name:
        try:
            with open(pickle_name, 'rb') as block_data:
                data = pickle.loads(block_data.read())
            print('reading binary of organized data')
            print('\n   ' + '\n   '.join([f'{name}:{ditem.shape}'
                       for ditem, name in zip(
                           list(data)[:4],
                           ['Xtrain', 'ytrain', 'Xtest', 'ytest'])]))
            if len(data) > 5:
                print(f'   is multi-class: {len(data) > 5} [{len(data[5])}-class]')
            print(f'   use 1-hot categorical feats: {data[4]}')
            return data
        except FileNotFoundError:
            pass
    df_meta = None
    if not(use_saved_meta_data):
        assert len(files_list) > 0, ('input file list is empty')
    else:
        df_meta = get_all_metadata()
        if len(files_list) == 0:
            files_list = sample_file_list()

    total_files = len(files_list)
    df, info = my_read_eeg_generic(
        (os.sep.join([prefix_directory, files_list[0]])
        if prefix_directory is not None else files_list[0]))
    nsamp, nsen = df.shape

    assert all([(sk == sck) for sk, sck in zip(senlist_known,
        df.columns.levels[df.columns.names.index('sensor')])]), (
            "sensor columns are out of order, need to create a sensor index "
            "sorting operation here...")

    count_train = count_test = 0
    for file in files_list:
        if 'test' in str.lower(file):
            count_test += 1
        else:
            # assuming we only have test and train files
            count_train += 1

    if multiclass:
        if debug > 1:
            print('multi-class labeling processing, reading header/meta data...')
        class_from_subject_id = GetClassId()
        if df_meta is None:
            for file in tqdm.tqdm(files_list):
                orig_tt_indic = (('test' in str.lower(file)) if pre_split else None)
                df, info = my_read_eeg_generic(
                    (os.sep.join([prefix_directory, files_list[0]])
                    if prefix_directory is not None else files_list[0]),
                    orig_tt_indic=orig_tt_indic, header_only=True)
                class_from_subject_id[info['subject']]
        else:
            meta_data_files_list = [f.split(os.sep)[-1] for f in df_meta.filename.values]
            # print('preview md files list', meta_data_files_list[:10])
            for file in tqdm.tqdm(files_list):
                # print('lookup_file', file.split(os.sep)[-1])
                index = meta_data_files_list.index(file.split(os.sep)[-1])
                # print(index)
                class_from_subject_id[df_meta.iloc[index]['subject']]
        enc_mult_id = OneHotEncoder()
        nclass = len(class_from_subject_id)
        if debug > 1:
            print(f'found {nclass} classes for multi-class id')
        enc_mult_id.fit([[subjid] for subjid in class_from_subject_id.keys()])
        if pre_split:
            ytrain = torch.zeros(count_train, nclass, dtype=torch.uint8)
            ytest = torch.zeros(count_test, nclass, dtype=torch.uint8)
        else:
            ylabels = torch.zeros(total_files, nclass, dtype=torch.uint8)
    else:
        if pre_split:
            ytrain = torch.zeros(count_train, dtype=torch.int)
            ytest = torch.zeros(count_test, dtype=torch.int)
        else:
            ylabels = torch.zeros(total_files, dtype=torch.int)

    if pre_split:
        Xtrain = torch.zeros(count_train, 1, nsamp + int(use_onehot_cat_feat), nsen, dtype=torch.float)
        Xtest = torch.zeros(count_test, 1, nsamp + int(use_onehot_cat_feat), nsen, dtype=torch.float)
    else:
        Xdata = torch.zeros(total_files, 1, nsamp + int(use_onehot_cat_feat), nsen, dtype=torch.float)
        simple_class_id = np.zeros(total_files, dtype=np.uint8)

    if use_onehot_cat_feat:
        enc_cat_features = OneHotEncoder()
        ncat_feat = len(match_types)
        enc_cat_features.fit([[mt] for mt in match_types])
        # cat_feat_list = []
        # cat_feat_trn = []
        # cat_feat_tst = []

    tst = trn = 0
    if debug >= 1:
        iterator = tqdm.tqdm(enumerate(files_list), total=total_files)
        iterator.set_description('reading in real-valued data...')
    else:
        iterator = enumerate(files_list)
    for fidx, file in iterator:
        orig_tt_indic = (('test' in str.lower(file)) if pre_split else None)

        if prefix_directory:
            full_url = os.sep.join([prefix_directory, file])
        else:
            full_url = file
        df, info = my_read_eeg_generic(full_url,
            orig_tt_indic=orig_tt_indic)
        if debug > 2:
            print(info)
        x, Z, xl, yl = process_data(df.index.values, df.values, 'time (s)', 'voltage (uV)')
        if pre_split:
            if orig_tt_indic:
                Xtest[tst, :, :nsamp, :] = torch.from_numpy(Z)
                if multiclass:
                    ytest[tst, :] = torch.from_numpy(enc_mult_id.transform(info['subject']))
                else:
                    ytest[tst] = int(info['alcoholic'])
                if use_onehot_cat_feat:
                    # cat_feat_tst.append(info['match'])
                    enc_feat = enc_cat_features.transform([[info['match']]]).toarray()
                    Xtest[tst, :, nsamp, :ncat_feat] = torch.from_numpy(enc_feat)
                tst += 1
            else:
                Xtrain[trn, :, :nsamp, :] = torch.from_numpy(Z)
                if multiclass:
                    ytrain[trn, :] = torch.from_numpy(enc_mult_id.transform(info['subject']))
                else:
                    ytrain[trn] = int(info['alcoholic'])
                if use_onehot_cat_feat:
                    # cat_feat_trn.append(info['match'])
                    enc_feat = enc_cat_features.transform([[info['match']]]).toarray()
                    Xtrain[trn, :, nsamp, :ncat_feat] = torch.from_numpy(enc_feat)
                trn += 1
        else:
            Xdata[fidx, :, :nsamp, :] = torch.from_numpy(Z)
            if multiclass:
                ylabels[fidx, :] = torch.from_numpy(enc_mult_id.transform(info['subject']))
                simple_class_id[fidx] = class_from_subject_id[info['subject']]
            else:
                ylabels[fidx] = simple_class_id[fidx] = int(info['alcoholic'])

            if use_onehot_cat_feat:
                # cat_feat_list.append(info['match'])
                enc_feat = enc_cat_features.transform([[info['match']]]).toarray()
                Xdata[fidx, :, nsamp, :ncat_feat] = torch.from_numpy(enc_feat)  # .reshape(1, 1, 1, 3)

    # if use_onehot_cat_feat:
    #     if pre_split:
    #         enc_feat = enc_cat_features.transform([[match_type] for match_type in cat_feat_trn])
    #         Xtrain[:, 0, nsamp, nsen:(nsen + ncat_feat)] = torch.from_numpy(enc_feat)
    #         enc_feat = enc_cat_features.transform([[match_type] for match_type in cat_feat_tst])
    #         Xtest[:, 0, nsamp, nsen:(nsen + ncat_feat)] = torch.from_numpy(enc_feat)
    #     else:
    #         enc_feat = enc_cat_features.transform([[match_type] for match_type in cat_feat_list])
    #         Xdata[:, 0, nsamp, nsen:(nsen + ncat_feat)] = torch.from_numpy(enc_feat)

    if not(pre_split):
        ss_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        nsplits_verify = 0
        for train_index, test_index in ss_split.split(Xtrain, simple_class_id):
            Xtrain, ytrain = Xdata[train_index], ylabels[train_index]
            Xtest, ytest = Xdata[test_index], ylabels[test_index]
            nsplits_verify += 1
            assert nsplits_verify <= 1, 'nsplits should only be 1'

    if debug:
        print('\n   ' + '\n   '.join([f'{name}:{ditem.shape}'
                   for ditem, name in zip(
                       list((Xtrain, ytrain, Xtest, ytest)),
                       ['Xtrain', 'ytrain', 'Xtest', 'ytest'])]))
    if multiclass:
        if pickle_name:
            print('saving binary of organized data')
            with open(pickle_name, 'wb') as block_data:
                block_data.write(pickle.dumps((Xtrain, ytrain, Xtest, ytest,
                    use_onehot_cat_feat, class_from_subject_id)))
        return Xtrain, ytrain, Xtest, ytest, use_onehot_cat_feat, class_from_subject_id
    else:
        if pickle_name:
            print('saving binary of organized data')
            with open(pickle_name, 'wb') as block_data:
                block_data.write(pickle.dumps((Xtrain, ytrain, Xtest, ytest,
                    use_onehot_cat_feat)))
        return Xtrain, ytrain, Xtest, ytest, use_onehot_cat_feat


def zero():
    """ functional zero for working with defaultdicts """
    return 0


def get_all_metadata(data_dirs=None, metadata_file_name=default_metadata_filen,
        df_type='wide', debug=1):

    if metadata_file_name and os.path.isfile(metadata_file_name):
        df = pd.read_csv(metadata_file_name)
        print(f"local metadata file found: {os.path.abspath(metadata_file_name)}")
        return df
    assert data_dirs is not None, ('could not find metadata file and data '
        'directories not defined')
    total_files = 0
    for orig_data_dir, directory in enumerate(data_dirs):
        for subject_dir in os.listdir(directory):
            if subject_dir in files_skip_processing:
                continue
            local_files = [f
                for f in os.listdir(os.sep.join([directory, subject_dir]))
                if f not in files_skip_processing]
            total_files += len(local_files)
    file_count = 0
    df = pd.DataFrame({})
    progress_bar = tqdm.tqdm(total=total_files, miniters=1)
    for directory in data_dirs:
        orig_data_dir = int(('test' in str.lower(directory)))
        for subject_dir in os.listdir(directory):
            if subject_dir in files_skip_processing:
                continue
            for file in os.listdir(os.sep.join([directory, subject_dir])):
                if file in files_skip_processing:
                    continue
                if debug == 1:
                    progress_bar.n = file_count
                    progress_bar.set_description('accumulating file metadata')

                full_file_url = os.sep.join([directory, subject_dir, file])
                if debug > 2:
                    print(f'read file header: {full_file_url}')

                df_file, info = my_read_eeg_generic(full_file_url,
                    df_type=df_type, orig_tt_indic=orig_data_dir,
                    header_only=True)
                info['filename'] = os.path.abspath(full_file_url)
                file_count += 1
                # if df is None:
                #     df = pd.DataFrame({k: [] for k in info})
                df = df.append(pd.DataFrame({k: [info[k]] for k in info}),
                    ignore_index=True)
    if metadata_file_name is not None:
        df.to_csv(metadata_file_name, index=False)
    return df


def sample_file_list(metadata_file_name=default_metadata_filen, limitby=None,
        limit_mult_files=None, balance_types=None, df_type='wide',
        shuffle_limited=False, seed=None, debug=1):
    """ select a (non-)random sample of the data according to the provided
        constraints
    """

    balance_types_bool = (balance_types is not None)
    if balance_types_bool:
        balance_type_count = defaultdict(zero)

    filter_files = (limitby is not None) or (balance_types_bool and (
        limit_mult_files is not None))

    if metadata_file_name and os.path.isfile(metadata_file_name):
        df = pd.read_csv(metadata_file_name)
    else:
        raise FileNotFoundError('could not fine metadata file')

    if not(filter_files):
        if limit_mult_files:
            return df.sample(n=None,
                random_state=seed).filename.values
        else:
            return df.filename.values
    else:
        if debug > 1:
            print('filter_settings:')
            if limit_mult_files:
                print(f'   limit_mult_files[{limit_mult_files}]')
            if balance_types_bool:
                print(f'   balance_types[{balance_types}]')
            if limitby:
                print(f'   limitby[{limitby}]')

            print(f'initial len df: {df.shape}')
        if limitby:
            for k, v in limitby.items():
                df = df[df[k].values == v]
            if debug > 1:
                print(f'after limitby len df: {df.shape}')
        if not(balance_types_bool):
            if limit_mult_files:
                return df.sample(n=limit_mult_files,
                    random_state=seed).filename.values
            else:
                return df.filename.values

        if isinstance(balance_types, list):
            balance_types = balance_types
        else:
            balance_types = [balance_types]

        if shuffle_limited:
            df_iter = df.sample(n=df.shape[0],
                random_state=seed)
        else:
            df_iter = df
        file_list = []
        file_count = 0
        if debug == 1:
            progress_bar = tqdm.tqdm(total=limit_mult_files, miniters=1)
        for ridx, row in df_iter.iterrows():
            skip_file = False
            for bt in balance_types:
                # print('bt', bt, row)
                key = bt[0]+str(row[bt[0]])
                if (balance_type_count[key] <
                        limit_mult_files/bt[1]):
                    balance_type_count[key] += 1
                else:
                    skip_file = True
                    break
            if skip_file:
                continue
            file_list.append(row['filename'])
            file_count += 1
            if debug == 1:
                progress_bar.n = file_count
                progress_bar.set_description('sampling file list')
            if file_count >= limit_mult_files:
                break
        return file_list


def accumulate_subject_file_list(data_dirs, limitby=None,
        limit_mult_files=np.inf, df_type='wide', debug=1):
    balance_types_bool = isinstance(limit_mult_files, tuple)
    if balance_types_bool:
        limit_mult_files, balance_types = limit_mult_files
        balance_type_count = defaultdict(zero)

    filter_files = limitby is not None or not(np.isinf(limit_mult_files[0])
            if balance_types_bool else np.isinf(limit_mult_files))

    total_files = 0
    files_list = []

    for orig_data_dir, directory in enumerate(data_dirs):
        for subject_dir in os.listdir(directory):
            if subject_dir in files_skip_processing:
                continue
            local_files = [os.sep.join([directory, subject_dir, f])
                for f in os.listdir(os.sep.join([directory, subject_dir]))
                if f not in files_skip_processing]
            total_files += len(local_files)
            if not(filter_files):
                files_list.extend(local_files)
    if debug == 1:
        # try:
        #     progress_bar = tqdm.tqdm_notebook(
        #         total=min(total_files, limit_mult_files),
        #         miniters=1)
        # except:
        progress_bar = tqdm.tqdm(
            total=min(total_files, limit_mult_files),
            miniters=1)
    file_count = 0
    unique_entries = defaultdict(set)
    if filter_files:
        filter_by_test_set = balance_types_bool and any(
            ['is_test_set' in bt[0] for bt in balance_types])
        if balance_types_bool:
            if isinstance(balance_types, list):
                balance_types = balance_types
            else:
                balance_types = [balance_types]
            if limitby:
                assert not(any([(bt[0] in limitby) for bt in balance_types])), (
                    "cannot balance a type key for which the data is also being "
                    "limited to in 'limitby'")
        if filter_by_test_set:
            filter_by_test_set = [bt[1] for bt in balance_types
                if 'is_test_set' in bt[0]][0]
        for directory in data_dirs:
            orig_data_dir = int(('test' in str.lower(directory)))
            for subject_dir in os.listdir(directory):
                if subject_dir in files_skip_processing:
                    continue
                if filter_by_test_set and orig_data_dir and (
                        balance_type_count['is_test_set' + str(orig_data_dir)] >=
                        limit_mult_files/filter_by_test_set):
                    continue
                for file in os.listdir(os.sep.join([directory, subject_dir])):
                    if file in files_skip_processing:
                        continue
                    if debug == 1:
                        progress_bar.n = file_count
                        progress_bar.set_description('accumulating file list')
                    if file_count >= limit_mult_files:
                        break

                    full_file_url = os.sep.join([directory, subject_dir, file])
                    if debug > 2:
                        print(f'read file header: {full_file_url}')

                    df, info = my_read_eeg_generic(full_file_url, df_type=df_type,
                         orig_tt_indic=orig_data_dir, header_only=True)

                    if limitby:
                        next_file = False
                        for k, v in limitby.items():
                            if v != info[k]:
                                next_file = True
                        if next_file:
                            continue
                    if (balance_types_bool and not(np.isinf(limit_mult_files))):
                        skip_file = False
                        for bt in balance_types:
                            key = bt[0]+str(info[bt[0]])
                            if (balance_type_count[key] <
                                    limit_mult_files/bt[1]):
                                balance_type_count[key] += 1
                            else:
                                skip_file = True
                                break
                        if skip_file:
                            continue
                    files_list.append(full_file_url)
                    for k in info:
                        unique_entries[k].add(info[k])
                    file_count += 1
    return files_list, unique_entries, len(files_list)


def my_read_eeg_generic(full_file_url, orig_tt_indic=None, header_only=False,
        df_type='wide'):
    """ wrap the file reader around parsing of different formats
        return dataframe and info
    """
    if full_file_url.endswith('gz'):
        with gzip.GzipFile(full_file_url, 'rb') as gzipfile:
            df = import_eeg_file(gzipfile, df_type=df_type,
                 orig_tt_indic=orig_tt_indic)
    #         elif full_file_url.endswith('bz2'):
    #             with open(full_file_url, 'rb') as input_file:
    #                 df = import_eeg_file(bz2.decompress(input_file.read()))
    else:
        # raw token stream never has any non-ASCII characters
        with open(full_file_url, 'r') as input_file:
            df = import_eeg_file(input_file, df_type=df_type,
                 orig_tt_indic=orig_tt_indic, header_only=header_only)
    info = {n: v for n, v in zip(
            df.columns.names, list(df.items())[0][0])
        if 'sensor' not in n}
    return df, info


def import_eeg_file(file_obj, df_type='long', optimize=True,
        orig_tt_indic=None, header_only=False):
    """
    Imports a file for a single EEG file and returns a wide or long dataframe.

    Parameters
    ----------
    file_obj
        A file-like object, such as a GzipFile, or a TextIOWrapper, or a
        regular file (such as from `open(<filename>)`)

    df_type : str, opt
        'long' or 'wide'.  If you want a 'long' dataframe or a 'wide' dataframe
        as an output.

    optimize: bool, opt
        True if you want data types to be coerced into their minimum sizes,
        false if you don't.

    Returns
    -------
    pandas.DataFrame
        The data from this file in a DataFrame object.
    """

    def parse_subject(line):
        return line[2:-4]

    def parse_alcoholic(line):
        char = line.strip('# ')[3]
        return True if char == 'a' else False

    def parse_obj(line):
        char = line.strip('# ')[1]
        return True if char == '1' else False

    def parse_match(line):
        string = line.strip('# ').split(',')[0].split(' ')[1]
        if string == 'nomatch':
            return 'nomatch'
        elif string == 'obj':
            return 'obj'
        elif string == 'match':
            return 'match'

    def parse_err(line):
        try:
            string = line.strip('# ').split(',')[0].split(' ')[2]
            if string == 'err':
                return True
            else:
                return False
        except IndexError as ie:
            return False
            # split_on_comma = line.strip('# ').split(',')
            # print("line.strip('# ').split(',')", split_on_comma)
            # print('take[0].split(' ')', split_on_comma[0].split(' '))
            # print('want index 2, len() = ', len(split_on_comma[0].split(' ')))
            # raise ie

    from io import TextIOWrapper
    if isinstance(file_obj, TextIOWrapper):
        text_obj = file_obj
    else:
        text_obj = TextIOWrapper(file_obj)

    header = []
    loc = None
    while True:
        loc = text_obj.tell()
        newline = text_obj.readline()
        if newline[0] == "#":
            header += [newline]
        else:
            text_obj.seek(loc)
            break

    subject = parse_subject(header[0])
    alcoholic = parse_alcoholic(header[0])
    obj = parse_obj(header[3])
    match = parse_match(header[3])
    err = parse_err(header[3])
    if not(header_only):
        df = pd.read_csv(text_obj, sep=' ', header=None,
            names=['trial', 'sensor', 'sample', 'value'], comment='#')
    else:
        df = pd.DataFrame({})
    df['alcoholic'] = alcoholic
    df['object'] = obj
    df['match'] = match
    df['err'] = err
    df['subject'] = subject
    if orig_tt_indic is not None:
        df['is_test_set'] = orig_tt_indic
    if header_only:
        return df

    if orig_tt_indic is not None:
        df = df[['subject', 'trial', 'is_test_set', 'alcoholic', 'match',
            'err', 'sensor', 'sample', 'value']]
    else:
        df = df[['subject', 'trial', 'alcoholic', 'match', 'err', 'sensor',
            'sample', 'value']]

    if optimize:
        df[['trial', 'sample']] = df[['trial', 'sample']].apply(
            pd.to_numeric, downcast='unsigned')
        df['value'] = df['value'].astype(np.float32)
        df['sensor'] = pd.Categorical(df['sensor'])
        df['match'] = pd.Categorical(df['match'])
        df['subject'] = pd.Categorical(df['subject'])
        if orig_tt_indic is not None:
            df['is_test_set'] = pd.Categorical(df['is_test_set'])

    if df_type == 'wide':
        if orig_tt_indic is not None:
            df = df.pivot_table(values='value', index='sample',
                columns=['subject', 'trial', 'is_test_set', 'alcoholic',
                    'match', 'err', 'sensor'])
        else:
            df = df.pivot_table(values='value', index='sample',
                columns=['subject', 'trial', 'alcoholic', 'match', 'err', 'sensor'])

    if df_type == 'long':
        if orig_tt_indic is not None:
            df = df.set_index(['subject', 'trial', 'is_test_set', 'alcoholic',
                'match', 'err', 'sample'])
        else:
            df = df.set_index(['subject', 'trial', 'alcoholic', 'match',
                'err', 'sample'])

    return df


eeg_half_half_alcoholic_all_subjects_obj_40 = [
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.018.gz']

eeg_half_half_alcoholic_all_subjects_match_100 = [
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.071.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.055.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.075.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.065.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.055.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.003.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.017.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.003.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.065.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.051.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.003.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.017.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.076.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.042.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.003.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.066.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.080.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.079.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.077.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.081.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.017.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.003.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.003.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.017.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.055.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.071.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.065.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.011.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.025.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.051.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.017.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.023.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.041.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.009.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.047.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.027.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.037.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.023.gz']


eeg_half_half_alcoholic_all_subjects_nomatch_100 = [
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.031.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.021.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.021.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.031.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.035.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.045.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.031.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.031.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.021.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.021.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.031.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.007.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.013.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.039.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.013.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.013.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.043.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.039.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.049.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.035.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.021.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.013.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.007.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.007.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.084.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.058.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.059.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.049.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.053.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.007.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.013.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.007.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.013.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.073.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.043.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.067.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.053.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.035.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.039.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.035.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.001.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.005.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.021.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.015.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.043.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.033.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.049.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.029.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.039.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.049.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.059.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.019.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.039.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.057.gz']


eeg_half_half_alcoholic_all_subjects_obj_100 = [
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.010.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.014.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.008.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.010.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.020.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.014.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.030.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.024.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.026.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.014.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.020.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.010.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.014.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.010.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.008.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.008.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.032.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.036.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.014.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.010.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.008.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.026.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.032.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.026.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.008.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.028.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.008.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.022.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.032.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.026.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.036.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.030.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.020.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.034.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.004.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.014.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.000.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.010.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.002.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.022.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.016.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.006.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.026.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.028.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.018.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.022.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.012.gz',
 f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.002.gz']

eeg_half_half_alcoholic_all_subjects_obj_tt_even_400 = [
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000372/co2a0000372.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000375/co2a0000375.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000344/co2c0000344.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000342/co2c0000342.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000345/co2c0000345.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000368/co2a0000368.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000369/co2a0000369.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000370/co2a0000370.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000377/co2a0000377.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000378/co2a0000378.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000371/co2a0000371.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000340/co2c0000340.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000347/co2c0000347.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000346/co2c0000346.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000341/co2c0000341.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000337/co2c0000337.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000339/co2c0000339.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2c0000338/co2c0000338.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.016.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.006.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.008.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000365/co2a0000365.rd.004.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.018.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.012.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.002.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.010.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.000.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TRAIN/co2a0000364/co2a0000364.rd.014.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000372/co2a0000372.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000375/co2a0000375.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.050.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.052.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000344/co2c0000344.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000342/co2c0000342.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000345/co2c0000345.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000368/co2a0000368.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000369/co2a0000369.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000370/co2a0000370.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.052.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.056.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.058.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.054.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.050.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000377/co2a0000377.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000378/co2a0000378.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000371/co2a0000371.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.037.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.043.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.047.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.045.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.041.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000340/co2c0000340.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.058.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.062.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.054.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000347/co2c0000347.rd.060.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.052.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.050.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000346/co2c0000346.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.024.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000341/co2c0000341.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.066.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.052.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.068.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.054.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000337/co2c0000337.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.050.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.054.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.060.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.058.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.056.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000339/co2c0000339.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.020.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.026.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.022.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2c0000338/co2c0000338.rd.028.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.042.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000365/co2a0000365.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.038.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.048.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.032.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.036.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.046.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.034.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.030.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.044.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.040.gz',
  f'./{large_dataset_prefix}/SMNI_CMI_TEST/co2a0000364/co2a0000364.rd.050.gz']
