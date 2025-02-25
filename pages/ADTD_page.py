####################################################
#
# ADTD page for DeconomiX GUI application
#
####################################################

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, _dash_renderer, dcc, html
import pandas as pd
from dash import dcc
from dash import html
import plotly.graph_objects as go
from utils.DeconomixCache import DCXCache
import numpy as np
import matplotlib
from matplotlib.colors import CenteredNorm

def get_adtd_layout(applCheckEnabled, localDCXCache):

    combobox_datasets_items = [
        {'value': 'train', 'label': "Training"},
        {'value': 'test', 'label': "Testing"}
    ]
    if localDCXCache.DeconomixFile.Application is not None:
        combobox_datasets_items.append({'value': 'appl', 'label': "Application"})

    adtd_layout = dmc.Stack(
        [
            html.Div(id="adtd-notify-container"),
            # Loading
            dmc.LoadingOverlay(
                visible=False,
                id="adtd-exec-overlay",
                loaderProps={"type": "oval", "color": "red", "size": "lg"},
                overlayProps={"radius": "sm", "blur": 2}
            ),

            # Parameters
            dmc.Fieldset(
                legend="Parameters",
                children=[
                    dmc.Group([
                        dmc.Fieldset(
                            children=[
                                dmc.Group([
                                    dmc.Checkbox(id="adtd-par-check-Cstatic", label="C static", labelPosition='left',
                                                 mb=10, checked=False),
                                    dmc.Checkbox(id="adtd-par-check-Deltastatic", label="Delta static",
                                                 labelPosition='left', mb=10, checked=False),
                                ]),

                                dmc.NumberInput(id="adtd-par-lambda1", label="Lambda 1", value=1e-6,
                                                min=1e-25, allowDecimal=True, allowNegative=False),
                                dmc.NumberInput(id="adtd-par-lambda2",
                                                label="Lambda 2", value=1e-6, min=1e-25,
                                                allowDecimal=True, allowNegative=False)
                            ],
                            legend="General parameters",
                            disabled=False,
                        ),
                        dmc.Fieldset(
                            children=[
                                dmc.NumberInput(id="adtd-par-lambda_min", label="Lambda 2 Minimum", value=1e-12,
                                                min=1e-25, allowDecimal=True, allowNegative=False),

                                dmc.NumberInput(id="adtd-par-lambda_max",
                                                label="Lambda 2 Maximum", value=1, min=1e-25,
                                                allowDecimal=True, allowNegative=False),

                                dmc.NumberInput(id="adtd-par-lambda_npoints",
                                                label="Lambda 2 N points", value=13, min=1,
                                                allowDecimal=False, allowNegative=False),
                            ],
                            legend="Hyperparameter Search",
                            disabled=False,
                        ),
                        dmc.Fieldset(
                            children=[
                                dmc.Select(label="Dataset",
                                           id="adtd-dataset-combo",
                                           value='test',
                                           data=combobox_datasets_items,
                                           allowDeselect=False),
                                dmc.NumberInput(id="adtd-par-iterations",
                                                label="Max iterations", value=100, min=1,
                                                allowDecimal=False, allowNegative=False),
                                dmc.Button("Run Lambda2 Search", id="adtd-run-hyper", fullWidth=True, mt=10),
                                dmc.Button("Execute ADTD", id="adtd-run", fullWidth=True, mt=10),
                            ],
                            legend="ADTD",
                            disabled=False,
                        )
                    ])

                ]
            ),

            # Results
            dmc.Skeleton(
                id="adtd-skeleton",
                visible=True,
                animate=False,
                children=[
                    dmc.Fieldset(
                        dmc.Tabs(
                            [
                                dmc.TabsList(
                                    [
                                        dmc.TabsTab("Hyperparameter Search", id='adtd-tab-hps', value="hps", disabled=True),
                                        dmc.TabsTab("Gene Regulation", id='adtd-tab-gr', value="gr"),
                                        dmc.TabsTab("Estimated Mixtures", id='adtd-tab-mix', value="mixtures")
                                    ]
                                ),
                                dmc.TabsPanel(value="hps",
                                              children=[
                                                  html.Div("Hyperparameter Search", id="adtd-res-hps")
                                              ]),

                                dmc.TabsPanel(value="gr",
                                              children=[
                                                  html.Div("Gene regulation plots", id="adtd-res-gr")
                                              ]),

                                dmc.TabsPanel(value="mixtures",
                                              children=[
                                                  html.Div("Estimated Mixture plots", id="adtd-res-mixtures")
                                              ])
                            ],
                            color='blue',
                            orientation='horizontal',
                            variant='default',
                            id="adtd-tab-panel",
                            value="mixtures"
                        )
                    )
                ]
            )
        ]
    )

    return adtd_layout


def getHPSPlot(localDCXCache):
    Losses = localDCXCache.ADTD_HPS_model.Losses

    avgLoss = Losses.mean(axis=0)
    stdLoss = Losses.std(axis=0)
    x_values = Losses.columns.to_list()

    fig = go.Figure()

    # Plot mean loss
    fig.add_trace(go.Scatter(
        x=x_values,
        y=avgLoss,
        mode='lines+markers',
        name='Mean Loss',
        line=dict(color='black', width=2),
        marker=dict(color='black')
    ))

    fig.add_trace(go.Scatter(
        x=x_values + x_values[::-1],
        y=(avgLoss + stdLoss).tolist() + (avgLoss - stdLoss).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(title='Lambda2', type='log', tickformat='.1e'),
        yaxis=dict(title='Loss', type='log', tickformat='.1e'),
        template='plotly_white'
    )

    return fig

def get_tab_adtd_mixture(localDCXCache, dataset, hidden):

    complete_dataset = dataset.copy()
    complete_dataset.loc['hidden'] = hidden.iloc[0]

    combobox_mixture_items = fill_combo_mixtures(complete_dataset)

    selected_data = localDCXCache.ADTDmodel.C_est.copy()
    selected_data.loc['hidden'] = localDCXCache.ADTDmodel.c_est.iloc[0]

    mixture = selected_data.iloc[:, 0]

    colors = dict(matplotlib.colors.cnames.items())
    hex_colors = tuple(colors.values())

    mixture_data = [
        {'name': mixture.index[i], "value": np.round(estimate, 4), "color": hex_colors[i + 25]} for i, estimate in
        enumerate(mixture)
    ]

    pie_plot = go.Figure(
        data=[
            go.Pie(
                labels=[item['name'] for item in mixture_data],
                values=[item['value'] for item in mixture_data],
                marker=dict(colors=[item['color'] for item in mixture_data]),
                textinfo="label+value",
                showlegend=False,
                hoverinfo="none",
                insidetextorientation="radial",
                textposition="outside"
            )
        ]
    )

    tab_adtd_mixture_layout = dmc.Stack(
        children=[
            dmc.Group(
                children=[
                    dmc.Select(label="Mixture",
                               id="adtd-mix-mixture-combo",
                               value='m1',
                               data=combobox_mixture_items,
                               allowDeselect=False),
                    dmc.Button("Download Estimates",
                                id="adtd-mix-dataset-button-download",
                               mt=25
                               ),
                    dcc.Download(id="adtd-mix-dataset-download")
                ]
            ),
            dmc.Group(
                children=[
                    dcc.Graph(id="adtd-mix-pie-plot",
                              figure=pie_plot,
                              responsive=True,
                              style = {"width": "100%", "height": "100%"})
                ]
            )
        ]
    )

    return tab_adtd_mixture_layout

def fill_combo_mixtures(dataset):
    nr_of_mixtures = len(dataset.columns)

    combobox_mixture_items = [
        {'value': f'm{i+1}', 'label': f"Mixture {i + 1}"} for i in range(0,nr_of_mixtures)
    ]

    return combobox_mixture_items

def get_tab_adtd_geneRegulation(localDCXCache, dataset):

    geneList = list(localDCXCache.DeconomixFile.X_mat.sort_index().index.unique())

    initially_selected = list(localDCXCache.ADTDmodel.Delta_est.abs().max(axis=1).sort_values(ascending=False).index.drop_duplicates())[:10]

    #initially_selected = geneList[0:10]

    gr_plot_init = get_gr_plot(localDCXCache, initially_selected)

    tab_adtd_gr = dmc.Stack(
        children=[
            dmc.Group(children=[
                dmc.MultiSelect(id="adtd-gr-selected-genes",
                                label="Select Genes to display",
                                data=geneList,
                                searchable=True,
                                nothingFoundMessage="Gene not found...",
                                hidePickedOptions=True,
                                value=initially_selected),
                dmc.Button("Download Gene Regulation",
                           id="adtd-gr-button-download",
                           mt=25
                           ),
                dcc.Download(id="adtd-gr-download")
            ]),

            dcc.Graph(id="adtd-gr-plot", figure=gr_plot_init)
        ]
    )

    return tab_adtd_gr


def get_gr_plot(localDCXCache, selected_genes):

    cell_types = list(localDCXCache.ADTDmodel.C_est.index.unique())
    delta = localDCXCache.ADTDmodel.Delta_est

    norm = CenteredNorm(1)
    normalized_delta = pd.DataFrame(norm(delta),
                                    index=delta.index,
                                    columns=delta.columns).loc[selected_genes, :]

    #normalized_delta = pd.DataFrame((delta.to_numpy() - np.min(delta.to_numpy())) / (np.max(delta.to_numpy()) - np.min(delta.to_numpy())),
    #                                index=delta.index,
    #                                columns=delta.columns).loc[selected_genes, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=normalized_delta,
            x=cell_types,
            y=selected_genes,
            colorscale="PuOr",
            colorbar=dict(), #title="Regulation Factor Δ"
            zmid=0.5
        )
    )

    fig.update_layout(
        xaxis_title="Cell Types",
        yaxis_title="Genes",
        xaxis=dict(tickmode="array", tickvals=list(range(len(cell_types))), ticktext=cell_types),
        yaxis=dict(tickmode="array", tickvals=list(range(len(selected_genes))), ticktext=selected_genes)
    )

    return fig
