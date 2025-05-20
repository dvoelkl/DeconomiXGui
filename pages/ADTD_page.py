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
import numpy as np
import matplotlib
from matplotlib.colors import CenteredNorm
from utils.global_cache import localDCXCache

def get_layout(checkApplEnabled=True):
    from utils.global_cache import localDCXCache
    return get_adtd_layout(checkApplEnabled)

def register_callbacks(app):
    from dash import Output, Input, State, no_update, ctx
    import dash_mantine_components as dmc
    from utils.DeconomixFile import DeconomixFile
    import numpy as np
    from pages.ADTD_page import getHPSPlot, get_tab_adtd_mixture, get_tab_adtd_geneRegulation, get_gr_plot
    import plotly.graph_objects as go
    import deconomix.methods
    # --- ADTD Callbacks ---
    # Callback for updating lambda1 input field
    @app.callback(
        Output('adtd-par-lambda1', 'disabled'),
        Input('adtd-par-check-Cstatic', 'checked'),
        prevent_initial_call=True
    )
    def updateLambda1Input(CstaticChecked):
        if CstaticChecked: return True
        else: return False

    # Callback for updating lambda2 input field
    @app.callback(
        Output('adtd-par-lambda2', 'disabled'),
        Input('adtd-par-check-Deltastatic', 'checked'),
        prevent_initial_call=True
    )
    def updateLambda2Input(DeltastaticChecked):
        if DeltastaticChecked: return True
        else: return False

    # Callback for running ADTD hyperparameter search
    @app.callback(
        Output('adtd-res-hps', 'children', allow_duplicate=True),
        Output('adtd-tab-panel', 'value', allow_duplicate=True),
        Output('adtd-tab-hps', 'disabled', allow_duplicate=True),
        Output('adtd-tab-gr', 'disabled', allow_duplicate=True),
        Output('adtd-tab-mix', 'disabled', allow_duplicate=True),
        Output('adtd-par-lambda2', 'value'),
        Input('adtd-run-hyper', 'n_clicks'),
        State('adtd-par-lambda_min', 'value'),
        State('adtd-par-lambda_max', 'value'),
        State('adtd-par-lambda_npoints', 'value'),
        State("adtd-dataset-combo", "value"),
        State('adtd-tab-gr', 'disabled'),
        State('adtd-tab-mix', 'disabled'),
        running=[(Output("adtd-skeleton", "visible"), True, False),
                 (Output("adtd-exec-overlay", "visible"), True, False)],
        prevent_initial_call=True
    )
    def runADTDHPS(n_clicks, lambda_min, lambda_max, npoints, dataset, gr_disabled, mix_disabled):
        localDCXCache.ADTD_config.lambda2min = lambda_min
        localDCXCache.ADTD_config.lambda2max = lambda_max
        localDCXCache.ADTD_config.nPoints = npoints
        localDCXCache.ADTD_config.Dataset = dataset
        if dataset == 'train':
            Y_hps = localDCXCache.DTD_Y_train
        elif dataset == 'appl':
            Y_hps = localDCXCache.DeconomixFile.Application.loc[localDCXCache.DTDmodel.gamma.index,:]
        else:
            Y_hps = localDCXCache.DTD_Y_test
        localDCXCache.ADTD_HPS_model = deconomix.methods.HPS(localDCXCache.DeconomixFile.X_mat.loc[Y_hps.index,:],
                                                             Y_hps,
                                                             localDCXCache.DTDmodel.gamma,
                                                             lambda_min=lambda_min,
                                                             lambda_max=lambda_max,
                                                             n_points=npoints)
        localDCXCache.ADTD_HPS_model.run()
        fig = getHPSPlot(localDCXCache)
        optimalLambda2 = localDCXCache.ADTD_HPS_model.lambda_max_gradient
        return dcc.Graph(figure=fig), 'hps', False, True, True, optimalLambda2

    # Callback for running ADTD
    @app.callback(
        Output('adtd-res-mixtures', 'children', allow_duplicate=True),
        Output('adtd-res-gr', 'children', allow_duplicate=True),
        Output('adtd-tab-panel', 'value', allow_duplicate=True),
        Output('adtd-tab-hps', 'disabled', allow_duplicate=True),
        Output('adtd-tab-gr', 'disabled', allow_duplicate=True),
        Output('adtd-tab-mix', 'disabled', allow_duplicate=True),
        Input('adtd-run', 'n_clicks'),
        State('adtd-par-check-Cstatic', 'checked'),
        State('adtd-par-check-Deltastatic', 'checked'),
        State('adtd-par-lambda1', 'value'),
        State('adtd-par-lambda2', 'value'),
        State("adtd-dataset-combo", "value"),
        State("adtd-par-iterations", "value"),
        State('adtd-tab-hps', 'disabled'),
        State('adtd-tab-gr', 'disabled'),
        State('adtd-tab-mix', 'disabled'),
        running=[(Output("adtd-skeleton", "visible"), True, False),
                 (Output("adtd-exec-overlay", "visible"), True, False)],
        prevent_initial_call=True
    )
    def runADTD(n_clicks, Cstatic, Deltastatic, lambda1, lambda2, dataset, nIter, hps_disabled, gr_disabled, mix_disabled):
        # Prüfe, ob Ergebnis bereits im Cache liegt
        cached = localDCXCache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)
        if cached is not None:
            return cached['tab_mixture'], cached['tab_gr'], 'mixtures', hps_disabled, Deltastatic, False
        previousDataset = localDCXCache.ADTD_config.Dataset
        localDCXCache.ADTD_config.Cstatic = Cstatic
        localDCXCache.ADTD_config.Deltastatic = Deltastatic
        localDCXCache.ADTD_config.lambda1 = lambda1
        localDCXCache.ADTD_config.lambda2 = lambda2
        localDCXCache.ADTD_config.Dataset = dataset
        localDCXCache.ADTD_config.nIter = nIter
        if dataset == 'train':
            Y_adtd = localDCXCache.DTD_Y_train
        elif dataset == 'appl':
            Y_adtd = localDCXCache.DeconomixFile.Application.loc[localDCXCache.DTDmodel.gamma.index,:]
        else:
            Y_adtd = localDCXCache.DTD_Y_test
        if not hps_disabled \
                and localDCXCache.ADTD_config.Dataset == previousDataset \
                and localDCXCache.ADTD_HPS_model is not None \
                and lambda2 == localDCXCache.ADTD_HPS_model.lambda_max_gradient:
            gamma = 1 / Y_adtd.shape[1] * np.ones(Y_adtd.shape[0]) / (Y_adtd.mean(axis=1))**2
        else:
            gamma = localDCXCache.DTDmodel.gamma
        localDCXCache.ADTDmodel = deconomix.methods.ADTD(localDCXCache.DeconomixFile.X_mat.loc[Y_adtd.index,:],
                                                         Y_adtd,
                                                         gamma,
                                                         lambda1,
                                                         lambda2,
                                                         nIter,
                                                         C_static=Cstatic,
                                                         Delta_static=Deltastatic)
        localDCXCache.ADTDmodel.run()
        tab_mixture = get_tab_adtd_mixture(localDCXCache, localDCXCache.ADTDmodel.C_est, localDCXCache.ADTDmodel.c_est)
        if not Deltastatic:
            tab_gr = get_tab_adtd_geneRegulation(localDCXCache, dataset)
        else:
            tab_gr = html.Div("Something went wrong, you shouldn't see this!")
        # Speichere Ergebnis im Cache
        localDCXCache.set_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic, {
            'tab_mixture': tab_mixture,
            'tab_gr': tab_gr,
            'C_est': localDCXCache.ADTDmodel.C_est.copy(),
            'c_est': localDCXCache.ADTDmodel.c_est.copy()
        })
        return tab_mixture, tab_gr, 'mixtures', hps_disabled, Deltastatic, False

    # Callback für Skeleton-Handling beim Dataset-Wechsel
    @app.callback(
        Output("adtd-skeleton", "visible"),
        Output('adtd-res-mixtures', 'children', allow_duplicate=True),
        Output('adtd-res-gr', 'children', allow_duplicate=True),
        Output('adtd-tab-panel', 'value', allow_duplicate=True),
        Output('adtd-tab-hps', 'disabled', allow_duplicate=True),
        Output('adtd-tab-gr', 'disabled', allow_duplicate=True),
        Output('adtd-tab-mix', 'disabled', allow_duplicate=True),
        Input("adtd-dataset-combo", "value"),
        State('adtd-par-check-Cstatic', 'checked'),
        State('adtd-par-check-Deltastatic', 'checked'),
        State('adtd-par-lambda1', 'value'),
        State('adtd-par-lambda2', 'value'),
        State("adtd-par-iterations", "value"),
        State('adtd-tab-hps', 'disabled'),
        State('adtd-tab-gr', 'disabled'),
        State('adtd-tab-mix', 'disabled'),
        prevent_initial_call=True
    )
    def restoreADTDResultsOnDatasetChange(dataset, Cstatic, Deltastatic, lambda1, lambda2, nIter, hps_disabled, gr_disabled, mix_disabled):
        # Setze das aktuelle Dataset im Cache, damit der Mechanismus konsistent bleibt
        localDCXCache.ADTD_config.Dataset = dataset
        cached = localDCXCache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)
        if cached is not None:
            return False, cached['tab_mixture'], cached['tab_gr'], 'mixtures', hps_disabled, Deltastatic, False
        else:
            return True, html.Div("Estimated Mixture plots", id="adtd-res-mixtures"), html.Div("Gene regulation plots", id="adtd-res-gr"), 'mixtures', True, True, True

    # Callback for updating ADTD pie plot
    @app.callback(
        Output("adtd-mix-pie-plot", "figure"),
        Input("adtd-mix-mixture-combo", "value"),
        State("adtd-dataset-combo", "value"),
        State('adtd-par-check-Cstatic', 'checked'),
        State('adtd-par-check-Deltastatic', 'checked'),
        State('adtd-par-lambda1', 'value'),
        State('adtd-par-lambda2', 'value'),
        State("adtd-par-iterations", "value"),
        prevent_initial_call=True
    )
    def UpdateADTDPiePlot(selectedMixture, dataset, Cstatic, Deltastatic, lambda1, lambda2, nIter):
        # Hole die korrekten Ergebnisse aus dem Cache für das aktuell gewählte Dataset und die aktuelle Parametrisierung
        cached = localDCXCache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)
        if cached is not None and 'C_est' in cached and 'c_est' in cached:
            C_est = cached['C_est'].copy()
            c_est = cached['c_est']
            C_est.loc['hidden'] = c_est.iloc[0]
            idx = int(selectedMixture[1:]) - 1
            mixture = C_est.iloc[:, idx]
            import matplotlib
            colors = dict(matplotlib.colors.cnames.items())
            hex_colors = tuple(colors.values())
            mixture_data = [
                {'name': mixture.index[i], "value": np.round(estimate, 4), "color": hex_colors[i + 25]} for i, estimate in enumerate(mixture)
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
            return pie_plot
        # Fallback: leeres Chart
        return go.Figure()

    # Callback for downloading ADTD estimated composition
    @app.callback(
        Output("adtd-mix-dataset-download", "data"),
        Input("adtd-mix-dataset-button-download", "n_clicks"),
        State("adtd-dataset-combo", "value")
    )
    def ADTDDownloadEstimate(n_clicks, selectedDataset):
        if "adtd-mix-dataset-button-download" == ctx.triggered_id:
            selected_data = localDCXCache.ADTDmodel.C_est.copy()
            selected_data.loc['hidden'] = localDCXCache.ADTDmodel.c_est.iloc[0]
            return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_ADTD_est.csv")
        else:
            return None

    # Callback for updating gene regulation plot
    @app.callback(
        Output("adtd-gr-selected-genes", "error"),
        Output("adtd-gr-plot", "figure"),
        Input("adtd-gr-selected-genes", "value"),
        State("adtd-gr-plot", "figure")
    )
    def UpdateGeneRegulationPlot(genes, curr_fig):
        error = ""
        fig = curr_fig
        if len(genes) < 1:
            error = "Select at least one gene!"
        else:
            fig = get_gr_plot(localDCXCache, sorted(genes, key=str.lower))
        return error, fig

    # Callback for downloading gene regulation results
    @app.callback(
        Output("adtd-gr-download", "data"),
        Input("adtd-gr-button-download", "n_clicks"),
        State("adtd-dataset-combo", "value")
    )
    def ADTDDownloadGR(n_clicks, selectedDataset):
        if "adtd-gr-button-download" == ctx.triggered_id:
            selected_data = localDCXCache.ADTDmodel.Delta_est
            return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_Delta.csv")
        else:
            return None

    # Callback for storing current ADTD tab in cache
    @app.callback(
        Input('adtd-skeleton', 'visible'),
        State("main-content", "children")
    )
    def storeCurrentADTDTab(skeletonVisible, currADTDTab):
        localDCXCache.ADTDTab = currADTDTab

def get_adtd_layout(applCheckEnabled):
    from utils.global_cache import localDCXCache
    # Robust None-check for cache and file
    if getattr(localDCXCache, 'DeconomixFile', None) is None:
        return dmc.Stack([
            dmc.Alert(
                title="No file loaded!",
                color="red",
                children=[
                    "Please upload a file first before using the ADTD page."
                ],
                style={"marginTop": 40, "marginBottom": 40}
            )
        ])

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
