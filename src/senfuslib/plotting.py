import logging
from typing import Any, Callable, List, Optional, Sequence, Union
from matplotlib import pyplot as plt
import numpy as np
from . import TimeSequence, MultiVarGauss, ConsistencyAnalysis, ConsistencyData
import itertools
import matplotlib as mpl

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, ColumnDataSource

FactoryType = Callable[[float, Any, float, Any, str, str], tuple[tuple, dict]]
t = np.linspace(0, np.pi*2, 100)
circle_points = np.stack((np.cos(t), np.sin(t)), axis=0)


def gauss_points(gauss: MultiVarGauss):
    return gauss.mean[:, None] + gauss.cholesky @ circle_points


def do_field(func: str, ax: plt.Axes, tseq: TimeSequence,
             y: Union[str, Sequence[str]] = None,
             fac: Optional[FactoryType] = None,
             x: str = None,
             **kwargs):
    if y is not None and not isinstance(y, (str, int)):
        for field in y:
            do_field(func, ax=ax, tseq=tseq, y=field, fac=fac, x=x, **kwargs)
        return

    if callable(fac):
        plots = dict()
        prev_list = None
        nextitems = itertools.islice(tseq.items(), 1, None)
        for (t, data), (t_next, data_next) in zip(tseq.items(), nextitems):
            data_out, plot_kwargs = fac(t=t, data=data, x=x, y=y,
                                        t_next=t_next, data_next=data_next)
            if data_out is None:
                continue

            current_list = plots.setdefault(tuple(plot_kwargs.items()), [])
            current_list.append(np.atleast_2d(data_out))

            add_nans = ['plot', 'fill_between']
            if (func in add_nans
                    and prev_list is not None
                    and current_list is not prev_list):
                prev_list.append(current_list[-1])
                prev_list.append(np.full_like(prev_list[-1], np.nan))
            prev_list = current_list

    else:
        plots = {tuple(): np.stack([
            tseq.field_as_array(x) if x else tseq.times,
            tseq.field_as_array(y)], axis=1)}

    for kwarg_tuple, data in plots.items():
        plot_kwargs = kwargs.copy()
        plot_kwargs.update(dict(kwarg_tuple))
        if label := plot_kwargs.pop('label', None):
            plot_kwargs['label'] = (label.replace('@y', str(y) or '')
                                    .replace('@x', str(x) or '')
                                    .replace('@', str(y) or ''))
        try:
            data = np.vstack([d for d in data if d.size]).swapaxes(0, 1)
            getattr(ax, func)(*data, **plot_kwargs)
        except ValueError:
            logging.warning("Could not plot data")


def plot_field(ax: plt.Axes, tseq: TimeSequence,
               y: Union[str, Sequence[str]] = None,
               fac: Optional[FactoryType] = None,
               x: str = None,
               **kwargs):
    do_field('plot', ax, tseq, y, fac, x, **kwargs)


def scatter_field(ax: plt.Axes, tseq: TimeSequence,
                  y: Union[str, Sequence[str]] = None,
                  fac: Optional[FactoryType] = None,
                  x: str = None,
                  **kwargs):
    do_field('scatter', ax, tseq, y, fac, x, **kwargs)


def fill_between_field(ax: plt.Axes, tseq: TimeSequence,
                       y: Union[str, Sequence[str]] = None,
                       fac: Optional[FactoryType] = None,
                       x: str = None,
                       **kwargs):
    assert fac is not None, "fill_between_field requires a factory"
    do_field('fill_between', ax, tseq, y, fac, x, **kwargs)


def ax_config(ax, x_label=None, y_label=None, title=None, aspect=None,
              legend=True, xlim=None, ylim=None, y_scale=None, x_scale=None):
    if x_label:
        assert (xlabl := ax.get_xlabel()) == '' or xlabl == x_label
        ax.set_xlabel(x_label)
    if y_label:
        assert (ylabl := ax.get_ylabel()) == '' or ylabl == y_label
        ax.set_ylabel(y_label)
    if aspect:
        ax.set_aspect(aspect)
    if legend:
        ax.legend(ncol=10)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if y_scale:
        ax.set_yscale(y_scale)
    if x_scale:
        ax.set_xscale(x_scale)


def fig_config(fig, window_title):
    try:
        fig.canvas.manager.set_window_title(window_title)
    except AttributeError:
        fig.canvas.set_window_title(window_title)

    fig.set_tight_layout(True)


def show_consistency(analysis: ConsistencyAnalysis,
                     fields_nis: Sequence[str] = tuple(),
                     fields_nees: Sequence[str] = tuple(),
                     fields_err: Sequence[str] = tuple(),
                     axs_nis: Sequence[plt.Axes] = tuple(),
                     axs_nees: Sequence[plt.Axes] = tuple(),
                     axs_err: Sequence[plt.Axes] = tuple(),
                     title: str = '',):
    all_axs = []

    def add_stuff(ax, data: ConsistencyData):
        all_axs.append(ax)
        sym = rf"$\chi^2_{{{data.dofs[0]}}}$"
        labels = [
            (f"{sym}, {data.in_interval:.0%}$\\in$"
             f"CI$_{{{data.alpha*100:.0f}\\%}}$"),
            f"{sym}, {data.above_median:.0%}>median",
            None]
        colors = ['tab:orange', 'tab:green', 'tab:orange']
        t = np.array(data.low_med_upp_tseq.times)
        lmu = data.low_med_upp_tseq.values_as_array()
        for i, (lab, col) in enumerate(zip(labels, colors)):
            ax.plot(t, lmu[:, i], ls='--', label=lab, color=col, alpha=0.7)

    for axs, fields, name in zip((axs_nis, axs_nees),
                                 (fields_nis, fields_nees),
                                 (f'NIS', f'NEES')):

        if not fields:
            continue
        # Check the size of axs to handle numpy arrays correctly
        if not hasattr(axs, 'size') or axs.size == 0:
            _, axs = plt.subplots(len(fields), 1, sharex=True)
        for ax, field in zip(axs, fields):
            if name == 'NIS':
                data = analysis.get_nis(field)
            else:
                data = analysis.get_nees(field)

            aconf = data.aconf
            insym = '$\\in$' if aconf[0] < data.a < aconf[1] else '$\\notin$'
            aconf = f'({aconf[0]:.3f}, {aconf[1]:.3f})'
            lab = f"{name}, A={data.a:.3f}{insym}{aconf}"
            ax.plot(data.mahal_dist_tseq.times,
                    data.mahal_dist_tseq.values, label=lab)
            add_stuff(ax, data)
            
            # Use a more descriptive label if the field is 'all'
            y_label = 'Total' if field == 'all' else f'{field}'
            ax_config(ax, y_label=y_label, y_scale='log')
            
        ax_config(axs[0], title=f'{name} {title}')
        ax_config(axs[-1], x_label='Time [s]')
        fig_config(axs[0].figure, f'{name} {title}')

    err_name = f'Error {title}'
    if fields_err and not axs_err:
        _, axs_err = plt.subplots(len(fields_err), 1, sharex=True)
    rmse_total = 0
    for ax, field in zip(axs_err, fields_err):
        err_gauss_tseq = analysis.get_x_err(field)
        err_tseq = err_gauss_tseq.map(lambda e: e.mean)
        std_tseq = err_gauss_tseq.map(lambda e: np.sqrt(e.cov).item())
        rmse = np.mean(np.array(err_tseq.values)**2)
        rmse_total += rmse
        ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.plot(err_tseq.times, err_tseq.values, label=f'err, rmse={rmse:.2e}')
        ax.fill_between(err_tseq.times, -np.array(std_tseq.values),
                        std_tseq.values,
                        alpha=0.5,
                        label=f'$\\pm 1\\sigma$')
        ax_config(ax, y_label=f'{field}')
    if fields_err:
        rmse_total /= len(fields_err)
        ax_config(axs_err[0], title=f'{err_name}, total rmse={rmse_total:.2e}')
        ax_config(axs_err[-1], x_label='Time [s]')
        fig_config(axs_err[0].figure, err_name)

    return all_axs

def interactive_show_consistency(
    analysis: ConsistencyAnalysis,
    fields_nis: Sequence[Union[str, List[str]]] = tuple(),
    fields_nees: Sequence[Union[str, List[str]]] = tuple(),
    fields_err: Sequence[str] = tuple(),
    title: str = '',
) -> gridplot:
    """
    Creates an interactive Bokeh plot to show consistency analysis results.
    """
    all_plots = []
    
    tooltips = [
        ("Name", "$name"),
        ("Timestep", "@timesteps"),
        ("Time", "@times{0.00}s"),
        ("Value", "@values{0.000}"),
    ]

    # --- 1. Process NIS and NEES plots ---
    for fields, name in zip((fields_nis, fields_nees), ('NIS', 'NEES')):
        if not fields:
            continue
        
        plots_group = []
        for i, field in enumerate(fields):
            data = analysis.get_nis(field) if name == 'NIS' else analysis.get_nees(field)
            
            plot_title = f'{name} {title}' if i == 0 else ''
            p = figure(title=plot_title, y_axis_type="log")
            
            times = data.mahal_dist_tseq.times
            values = data.mahal_dist_tseq.values
            timesteps = np.arange(len(values))
            source = ColumnDataSource(data={'timesteps': timesteps, 'times': times, 'values': values})

            aconf = data.aconf
            insym = '∈' if aconf[0] < data.a < aconf[1] else '∉'
            lab = f"{name}, A={data.a:.3f} {insym} ({aconf[0]:.3f}, {aconf[1]:.3f})"
            
            main_line_renderer = p.line(
                x='timesteps', y='values', source=source, 
                legend_label=lab, name=lab, color="royalblue", line_width=2
            )
            
            subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            dofs_sub = str(data.dofs[0]).translate(subscript_map)
            sym = f"χ²{dofs_sub}"
            
            ci_label = (f"{sym}, {data.in_interval:.0%} ∈ "
                        f"CI at {data.alpha*100:.0f}%")
            median_label = f"{sym}, {data.above_median:.0%} > median"
            
            lmu = data.low_med_upp_tseq.values_as_array()
            p.line(timesteps, lmu[:, 0], legend_label=ci_label, name="CI Bound", color="darkorange", line_dash='dashed')
            p.line(timesteps, lmu[:, 2], color="darkorange", line_dash='dashed')
            p.line(timesteps, lmu[:, 1], legend_label=median_label, name="Median", color="green", line_dash='dashed')
            
            hover_tool = HoverTool(
                renderers=[main_line_renderer], 
                tooltips=tooltips, 
                mode='vline', 
                point_policy='snap_to_data'
            )
            p.add_tools(hover_tool)

            if isinstance(field, (list, tuple)):
                # Convert all items to strings before joining
                label_text = ', '.join(map(str, field))
            else:
                label_text = str(field)

            p.yaxis.axis_label = label_text
            p.legend.location = "bottom_right"
            p.legend.click_policy = "hide"
            plots_group.append(p)

        if plots_group:
            plots_group[-1].xaxis.axis_label = 'Timestep'
            all_plots.extend(plots_group)

    # --- 2. Process Error plots ---
    if fields_err:
        err_plots = []
        rmse_total_sq = 0
        for i, field in enumerate(fields_err):
            err_gauss_tseq = analysis.get_x_err(field)
            err_tseq = err_gauss_tseq.map(lambda e: e.mean)
            std_tseq = err_gauss_tseq.map(lambda e: np.sqrt(e.cov).item())
            rmse = np.sqrt(np.mean(np.array(err_tseq.values)**2))
            rmse_total_sq += rmse**2
            
            plot_title = f'Error {title}' if i == 0 else ''
            p = figure(height=300, width=800, title=plot_title)

            times = err_tseq.times
            values = err_tseq.values
            timesteps = np.arange(len(values))
            source_err = ColumnDataSource(data={'timesteps': timesteps, 'times': times, 'values': values})

            err_label = f'err, rmse={rmse:.2e}'
            err_line_renderer = p.line(x='timesteps', y='values', source=source_err, legend_label=err_label, name=err_label, color='royalblue')
            
            p.varea(x=timesteps, y1=-np.array(std_tseq.values), y2=np.array(std_tseq.values), 
                    fill_alpha=0.3, fill_color='royalblue', legend_label='±1σ', name='±1σ')
            
            p.line([timesteps[0], timesteps[-1]], [0, 0], color='black', line_dash='dashed', alpha=0.7)
            
            hover_tool = HoverTool(
                renderers=[err_line_renderer], 
                tooltips=tooltips, 
                mode='vline', 
                point_policy='snap_to_data'
            )
            p.add_tools(hover_tool)

            p.yaxis.axis_label = field
            p.legend.location = "bottom_right"
            p.legend.click_policy = "hide"
            err_plots.append(p)
        
        if err_plots:
            total_rmse = np.sqrt(rmse_total_sq / len(fields_err))
            err_plots[0].title.text = f'Error {title}, total rmse={total_rmse:.2e}'
            err_plots[-1].xaxis.axis_label = 'Timestep'
            all_plots.extend(err_plots)

    # --- 3. Arrange all plots into a final grid layout ---
    if not all_plots:
        return None
        
    for plot in all_plots[1:]:
        plot.x_range = all_plots[0].x_range
        
    return gridplot(all_plots, ncols=1, sizing_mode='stretch_both')