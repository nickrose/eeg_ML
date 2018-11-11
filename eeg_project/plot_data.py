""" some tools for plotting EEG data and doing visual comparison """
from eeg_project.read_data import (my_read_eeg_generic, SAMP_FREQ,
    pass_through, accumulate_subject_file_list, files_skip_processing,
    sample_file_list, match_types)
import numpy as np
import pandas as pd
from collections import defaultdict
# from six import text_type
import tqdm
from matplotlib import pyplot, cm
# from mpl_toolkits.mplot3d import Axes3D
# # import ipywidgets as widgets
from IPython.display import clear_output


def highlight_correlated_feature_twoclass(
        file_samples=100,
        match_types_in=match_types, figsize=(12, 14),
        process_data=None, pca_vec_to_plot=5, debug=1):
    """ sub-sample the data set and plot correlation information """
    corr_accum = None
    if process_data is not None:
        data_type = '(frqeuency)'
    else:
        data_type = '(time-domain)'
    for match_type in match_types_in:
        for aidx, is_alcoholic in enumerate([True, False]):
            corr_accum = cov_accum = None
            if debug:
                print(f'getting example data for match_type[{match_type}]'
                    f' and is_alcoholic[{is_alcoholic}]')
            file_list = sample_file_list(
                    limitby=dict(
                        match=match_type,
                        alcoholic=is_alcoholic),
                    limit_mult_files=file_samples,
                    balance_types=[('subject', 10)], df_type='wide',
                    seed=42, debug=max(0, debug - 1))

            for file in tqdm.tqdm(file_list):
                df, info = my_read_eeg_generic(
                    file, orig_tt_indic=('test' in str.lower(file)))

                corr = df.corr()
                cov = df.cov().values
                if corr_accum is None:
                    sen_names = corr.columns.levels[
                        corr.columns.names.index('sensor')]
                    nsen = len(sen_names)
                if process_data:
                    x, Z, xl, yl = process_data([], df.values, '',
                        '', fs=SAMP_FREQ)
                    df = pd.DataFrame(Z)
                    corr = df.corr()
                    cov = df.cov().values
                if corr_accum is None:
                    corr_accum = corr.values
                    cov_accum = cov/nsen
                else:
                    corr_accum += corr.values
                    cov_accum += cov/nsen

            corr_accum /= len(file_list)
            if aidx == 0:
                corr_alcoholic = corr_accum.copy()
                cov_alcoholic = cov_accum.copy()
            else:
                corr_nonalch = corr_accum.copy()
                cov_nonalch = cov_accum.copy()

            if debug > 1:
                pyplot.figure(figsize=figsize)
                pyplot.pcolor(np.flipud(corr_accum))
                pyplot.xticks(np.arange(nsen), sen_names)
                pyplot.yticks(np.arange(nsen), reversed(sen_names))
                pyplot.title(f'corr - across sensors {data_type} - '
                    f'is_alcoholic[{is_alcoholic}] - match[{match_type}]')
                pyplot.colorbar()
                pyplot.show()

        Ua, svs_a, Va = np.linalg.svd(cov_alcoholic, full_matrices=False,
            compute_uv=True)
        Una, svs_na, Vna = np.linalg.svd(cov_nonalch, full_matrices=False,
            compute_uv=True)
        print('SVec size', Una.shape)
        # print(svs_a)
        pyplot.figure(figsize=(figsize[0], 6))
        leg = []
        lg, = pyplot.plot(svs_a, label='alcoholic')
        leg.append(lg)
        lg, = pyplot.plot(svs_na, label='not alcoholic')
        leg.append(lg)
        pyplot.legend(handles=leg)
        # pyplot.xticks(np.arange(nsen), sen_names)
        # pyplot.yticks(np.arange(nsen), reversed(sen_names))
        pyplot.title(f'PCA decomposition: SVs - across sensors {data_type} - '
            f'- match[{match_type}]')
        pyplot.show()

        pyplot.figure(figsize=(figsize[0]+4, 6))
        leg = []
        if isinstance(pca_vec_to_plot, tuple):
            pca_vec_to_plot_count = pca_vec_to_plot[1] - pca_vec_to_plot[0]
        else:
            pca_vec_to_plot_count = pca_vec_to_plot
            pca_vec_to_plot = (0, pca_vec_to_plot)
        pca_vec_to_plot_count = min(pca_vec_to_plot_count, nsen)
        for i, idx in enumerate(range(*pca_vec_to_plot)):
            if i == 0:
                clrdict = {}
            lg, = pyplot.plot(Ua[:, idx], linewidth=2*(pca_vec_to_plot_count - i + 1),
                label='alcoholic', **clrdict)
            if i == 0:
                leg.append(lg)
                clrdict = {'color': lg.get_color()}
        for i, idx in enumerate(range(*pca_vec_to_plot)):
            if i == 0:
                clrdict = {}
            lg, = pyplot.plot(Una[:, idx], linewidth=2*(pca_vec_to_plot_count - i + 1),
                label='not alcoholic', **clrdict)
            if i == 0:
                leg.append(lg)
                clrdict = {'color': lg.get_color()}

        pyplot.xticks(np.arange(nsen), sen_names)
        pyplot.legend(handles=leg)
        # pyplot.xticks(np.arange(nsen), sen_names)
        # pyplot.yticks(np.arange(nsen), reversed(sen_names))
        pyplot.title(f'PCA decomposition: first {pca_vec_to_plot} '
            f'singular vectors - across sensors {data_type} - '
            f'match[{match_type}]')

        pyplot.figure(figsize=figsize)
        pyplot.pcolor(np.flipud(corr_alcoholic - corr_nonalch))
        pyplot.xticks(np.arange(nsen), sen_names)
        pyplot.yticks(np.arange(nsen), reversed(sen_names))
        pyplot.title(f'corr - across sensors {data_type} - '
            f'(alcoholic-nonalcoholic) - match[{match_type}]')
        pyplot.colorbar()
        pyplot.show()
        return Ua[:, :pca_vec_to_plot], Una[:, :pca_vec_to_plot]


def plot_data_subject_dirs(data_dirs=None, file_list=None,
        labelby=None, limitby=None, plots=None, figsize=None,
        transparency=1., yscale='linear', xrange=None,
        yrange=None, force_axes_same_scale=True,
        process_data=pass_through, limit_mult_files=np.inf, debug=1):
    """ plot EEG data by searching subject directories with some options """
    df_type = 'wide'
    if plots is None:
        plots = dict(grid=True)
    senlistorder = None

    all_data_overlaid = ('all_data_traces' in plots and
            (plots['all_data_traces'] is not None))
    printed_entry_info = False
    if ((data_dirs is None) and (file_list is None)):
        if isinstance(limit_mult_files, tuple):
            limit_mult, bal_list = limit_mult_files
        else:
            bal_list = None
            limit_mult = limit_mult_files
        if np.isinf(limit_mult):
            limit_mult = None
        file_list = sample_file_list(
                limitby=limitby,
                limit_mult_files=limit_mult,
                balance_types=bal_list, df_type=df_type,
                seed=42, debug=max(0, debug - 1))

    if file_list is None:
        file_list, unique_entries, total_files = accumulate_subject_file_list(
            data_dirs, limitby=limitby, limit_mult_files=limit_mult_files,
            df_type=df_type, debug=debug)
        if debug:
            print('unique entries in metadata from file accumulation')
            for k in unique_entries:
                print(f'   {k}: {unique_entries[k]}')
            printed_entry_info = True
    else:
        total_files = len(file_list)
    plot_sensor = None
    if all_data_overlaid:
        if transparency == 1.:
            transparency = 0.5
        plot_to_make = sum([bool(plots[k]) for k in plots if 'all_data' not in k])
        if isinstance(plots['all_data_traces'], str):
            plot_sensor = plots['all_data_traces']
        if plot_to_make == 0:
            if isinstance(plots['all_data_traces'], str):
                plots['overlap'] = True
            else:
                plots['grid'] = True
        assert sum([bool(plots[k]) for k in plots if 'all_data' not in k]) == 1, (
            "cannot display multiple plot types")
        assert isinstance(plots['all_data_traces'], str) or (
            'overlap' not in plots or not(plots['overlap'])), (
                "cannot plot single overlapping plot if sensor is not specified")
        if figsize is None and 'grid' in plots and isinstance(plots['grid'], str):
            if plots['grid'].startswith('square'):
                figsize = (16, 18)
            else:
                figsize = (15, 64 * 8)

        pyplot.figure(figsize=figsize)
        legd = []
        running_min_max = (np.inf, -np.inf)
        if debug == 1:
            # try:
            #     progress_bar = tqdm.tqdm_notebook(total=total_files, miniters=1)
            # except:
            progress_bar = tqdm.tqdm(total=total_files, miniters=1)
    else:
        legd = None
        if figsize is None:
            figsize = (12, 14)
    if isinstance(limit_mult_files, tuple):
        limit_mult_files = limit_mult_files[0]

    file_count = 0
    color_dict = dict()
    unique_entries = defaultdict(set)
    for file in file_list:
        orig_data_dir = int(('test' in str.lower(file)))
        if file in files_skip_processing:
            continue
        if all_data_overlaid:
            if debug == 1:
                progress_bar.n = file_count
                progress_bar.set_description('files processed')
            if file_count >= limit_mult_files:
                break
        else:
            clear_output()

        full_file_url = file  # os.sep.join(file)
        if debug > 1:
            print(f'read file: {full_file_url}')

        df, info = my_read_eeg_generic(full_file_url, df_type=df_type,
             orig_tt_indic=orig_data_dir)

        if all_data_overlaid:
            if labelby and labelby in info:
                id = labelby + ':' + str(info[labelby])
            else:
                id = info['subject']
        else:
            id = None
        if debug > 1:
            print(' | '.join([f'{n:>8s}:{str(v):4s}' for n, v in info.items()]))

        sen_index = df.columns.names.index('sensor')
        senlist = df.columns.levels[sen_index]
        if senlistorder is None:
            senlistorder = senlist
        elif all_data_overlaid:
            assert all([sl == chkl
                for sl, chkl, in zip(senlist, senlistorder)]), (
                'different data set has list of sensors in a '
                'different order')
        Z = df.values
        nsamp, nsen = Z.shape
        time = np.arange(nsamp) / SAMP_FREQ
        x_data, Z, xlabel, ylabel = process_data(time, Z, 'time (s)',
            'voltage (uV)', fs=SAMP_FREQ)
        # nsamp, nsen = Z.shape
        if all_data_overlaid and force_axes_same_scale:
            running_min_max = (min(Z.min(), running_min_max[0]),
                max(Z.max(), running_min_max[1]))
            minv, maxv = running_min_max
        else:
            minv = maxv = None
        if ('overlap' in plots and plots['overlap']):
            plot_all_overlaid(x_data, Z, xlabel, ylabel, senlist, figsize,
                id=id, yscale=yscale, yrange=yrange, xrange=xrange,
                multi_trace_plot_labels=(file_count == 0),
                color_dict=color_dict, transparency=transparency,
                plot_sensor=plot_sensor, legend=legd)

        if ('grid' in plots and plots['grid']):
            grid_square = (not(isinstance(plots['grid'], str)) or
                    plots['grid'].startswith('square'))
            plot_grid(x_data, Z, xlabel, ylabel, senlist, minv, maxv,
                id=id, grid_square=grid_square, figsize=figsize,
                multi_trace_plot_labels=(file_count == 0),
                yscale=yscale, yrange=yrange, xrange=xrange,
                color_dict=color_dict, transparency=transparency,
                legend=legd)

        if ('threed' in plots and plots['threed']) and not(
                all_data_overlaid):
            y_data = df.columns.labels[sen_index].values()
            plot_3d(x_data, y_data, Z, df, xlabel, ylabel, figsize=figsize)

        if not(all_data_overlaid):
            input('press enter to cont...')
        file_count += 1
        for k in info:
            unique_entries[k].add(info[k])

        if file_count >= limit_mult_files:
            break

    if all_data_overlaid:
        if 'overlap' in plots and plots['overlap']:
            pyplot.xlabel(xlabel, fontsize=14)
            pyplot.ylabel(ylabel, fontsize=15)
            # if minmax[1]/(minmax[0] if minmax[0] > 0 else 1.) > 1e1:
        #             pyplot.axes().set_xscale('log', basex=2)
            pyplot.title(f'Sensor: {plots["all_data_traces"]}', fontsize=15)
        pyplot.legend(handles=legd, fontsize=15)  # , loc='lower right')
        pyplot.show()
    if debug and not(printed_entry_info):
        print('unique entries in metadata from file accumulation')
        for k in unique_entries:
            print(f'   {k}: {unique_entries[k]}')
    return file_list


def aggregate_behavior(Z):
    """ returns some basic trace information """
    nsamp, nsen = Z.shape
    median_trace = np.median(Z, axis=1)
    dev = np.std(Z - np.repeat(np.matrix(median_trace).transpose(),
        nsen, axis=1), axis=1)
    cmpr_high_variability = [(Z[:, sen_i] > median_trace + 2 * dev
        ).sum()/nsamp > 0.5 for sen_i in range(nsen)]
    return nsamp, nsen, cmpr_high_variability, median_trace, dev


def plot_grid(x_data, Z, xlabel, ylabel, senlist,
        minv=None, maxv=None, id=None, grid_square=True,
        figsize=(12, 15), multi_trace_plot_labels=False,
        yscale='linear', xrange=None, yrange=None,
        color_dict={}, transparency=1., legend=None):
    """ plot a gride of sensor traces """
    nsen = len(senlist)
    all_data_overlaid = (id is not None) and (legend is not None)
    grid_base_sz = int(np.ceil(np.sqrt(nsen)))
    # yscale = ('log' if 'Hz' in xlabel else 'linear')

    if grid_square:
        ncols = nrows = grid_base_sz
    else:
        ncols, nrows = 1, nsen
    sen_i = 0
    coli = rowi = 0
    if all_data_overlaid:
        pyplot.subplots_adjust(wspace=.2, hspace=.35)
        for sen_i, sen in enumerate(senlist):
            pyplot.subplot(nrows, ncols, sen_i+1)
            if id in color_dict:
                clrdict = {'color': color_dict[id]}
            else:
                clrdict = {}
            # print('sizes x_data, Z', x_data.shape, Z.shape)
            lg, = pyplot.plot(x_data, Z[:, sen_i],
                '-', label=id, alpha=transparency, **clrdict)
            if id not in color_dict:
                legend.append(lg)
                color_dict[id] = lg.get_color()
            if minv is not None and maxv is not None:
                pyplot.ylim((minv, maxv))
            if multi_trace_plot_labels:
                pyplot.title(sen, fontdict=dict(size=10))
                # pyplot.tick_params(axis='y', which='major', labelsize=7)
                if ncols == 1 or (coli == grid_base_sz//2 and
                        rowi == grid_base_sz):
                    pyplot.xlabel(xlabel)
                if ncols == 1 or (coli == 0 and
                        rowi == grid_base_sz//2):
                    pyplot.ylabel(ylabel)
                pyplot.grid(True)
                if yscale:
                    pyplot.yscale(yscale)
                if xrange:
                    pyplot.xlim(xrange)
                if yrange:
                    pyplot.ylim(yrange)
            sen_i += 1
            coli += 1
            if coli >= ncols:
                rowi += 1
                coli = 0
    else:
        minv, maxv = Z.min(), Z.max()
        nsamp, nsen, cmpr_high_variability, median_trace, dev = \
            aggregate_behavior(Z)
        pyplot.figure(figsize=figsize)
        pyplot.subplots_adjust(wspace=.2, hspace=.35)
        for sen_i, sen in enumerate(senlist):
            pyplot.subplot(nrows, ncols, sen_i+1)
            lg, = pyplot.plot(x_data, Z[:, sen_i],
                (':' if cmpr_high_variability else '-'),
                label=sen)
            pyplot.ylim((minv, maxv))
            pyplot.title(sen, fontdict=dict(size=10))
            pyplot.tick_params(axis='y', which='major', labelsize=7)
            if ncols == 1 or (coli == int(grid_base_sz/2) and
                    rowi == grid_base_sz - 1):
                pyplot.xlabel(xlabel)
            if ncols == 1 or (coli == 0 and
                    rowi == int(grid_base_sz/2)):
                pyplot.ylabel(ylabel)
            pyplot.grid(True)
            if yscale:
                pyplot.yscale(yscale)
            if xrange:
                pyplot.xlim(xrange)
            if yrange:
                pyplot.ylim(yrange)
            sen_i += 1
            coli += 1
            if coli >= ncols:
                rowi += 1
                coli = 0
        pyplot.show()


def plot_3d(x_data, y_data, Z, df, xlabel, ylabel, xrange=None,
            yrange=None, figsize=(12, 12)):
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        nsamp, nsen = Z.shape

        sen_index = df.columns.names.index('sensor')
        senlist = df.columns.levels[sen_index]
        pyplot.yticks(y_data, senlist)
        ax.plot_surface(
            np.repeat(x_data,
                nsen, axis=1),
            np.repeat(np.matrix(y_data), nsamp, axis=0),
            df.values,
            cmap=cm.coolwarm)  # , rcount=270, ccount=270)
        pyplot.xlabel(xlabel)
        pyplot.ylabel('Sensor name')
        # zscale = ('log' if 'Hz' in xlabel else 'linear')
        # pyplot.zscale(zscale)
        ax.set_zlabel(ylabel)
        ax.view_init(elev=45., azim=-130)
        ax.tick_params(axis='y', which='major', labelsize=4)
        pyplot.show()


def plot_all_overlaid(x_data, Z, xlabel, ylabel, sen_list, figsize=(12, 14),
        multi_trace_plot_labels=True,
        id=None, plot_sensor=None, yscale='linear', xrange=None,
        yrange=None, legend=None, color_dict={}, transparency=1.):
    """ plot some overlapping data for single subject/trial or for all data on
        a single sensor
    """
    all_data_overlaid = (plot_sensor is not None) and (legend is not None) and (
        id is not None)
    if not(all_data_overlaid):
        pyplot.figure(figsize=figsize)
        legend = []
    # yscale = ('log' if 'Hz' in xlabel else 'linear')
    for sen_i, sen in enumerate(sen_list):
        if all_data_overlaid:
            if sen == plot_sensor:
                if id in color_dict:
                    clrdict = {'color': color_dict[id]}
                else:
                    clrdict = {}
                lg, = pyplot.plot(x_data, Z[:, sen_i],
                    '-', label=id, alpha=transparency, **clrdict)
                if id not in color_dict:
                    legend.append(lg)
                    color_dict[id] = lg.get_color()
            else:
                continue
        else:
            nsamp, nsen, cmpr_high_variability, median_trace, dev = \
                aggregate_behavior(Z)
            lg, = pyplot.plot(x_data, Z[:, sen_i],
                (':' if (cmpr_high_variability and
                    cmpr_high_variability[sen_i]) else '-'),
                label=sen, alpha=transparency)
            legend.append(lg)
            if multi_trace_plot_labels and median_trace is not None:
                lg, = pyplot.plot(x_data, median_trace, '--',
                    label='median', linewidth=5)

#             legend.append(lg)
            pyplot.xlabel(xlabel, fontsize=14)
            pyplot.ylabel(ylabel, fontsize=15)
            # pyplot.yscale(yscale)
            # if minmax[1]/(minmax[0] if minmax[0] > 0 else 1.) > 1e1:
#             pyplot.axes().set_xscale('log', basex=2)
            pyplot.legend(handles=legend, fontsize=7)  # , loc='lower right')
            pyplot.title('Sensor traces', fontsize=15)

    if yscale:
        pyplot.yscale(yscale)
    if xrange:
        pyplot.xlim(xrange)
    if yrange:
        pyplot.ylim(yrange)
    if not(all_data_overlaid):
        pyplot.show()
