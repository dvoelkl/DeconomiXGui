####################################################
#
# DTD page for DeconomiX GUI application
#
####################################################

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, _dash_renderer, dcc, html
import pandas as pd
from dash import dcc
from dash import html
import plotly.graph_objects as go
import dash_bio as dashbio
import numpy as np

def get_layout(checkApplEnabled=True, n_genes=0):
    from utils.global_cache import localDCXCache
    return get_dtd_layout(checkApplEnabled, n_genes)

def register_callbacks(app):
    from dash import Output, Input, State, no_update, ctx
    import dash_mantine_components as dmc
    from utils.DeconomixFile import DeconomixFile
    import numpy as np
    from scipy.stats import spearmanr
    import plotly.graph_objects as go
    from pages.DTD_page import get_tab_dtd_loss, get_tab_dtd_correlation, get_tab_dtd_mixture, get_tab_dtd_markermap, get_correlation_dict, fill_combo_mixtures, get_markermap_plot
    from deconomix.utils import simulate_data, calculate_estimated_composition
    from deconomix.methods import DTD
    from utils.global_cache import localDCXCache

    @app.callback(
        Output('dtd-exec-overlay', 'visible', allow_duplicate=True),
        Output('dtd-skeleton', 'visible', allow_duplicate=True),
        State('dtd-par-training-n-mix', 'value'),
        State('dtd-par-training-n-cells-in-mix', 'value'),
        State('dtd-par-test-n-mix', 'value'),
        State('dtd-par-test-n-cells-in-mix', 'value'),
        State('dtd-par-check-ApplData', 'checked'),
        State('dtd-par-num-iter', 'value'),
        Input('dtd-run', 'n_clicks'),
        prevent_initial_call=True
    )
    def prepareDTD(train_n_mix, train_n_cells, test_n_mix, test_n_cell, fApplData, n_iter, n_clicks):
        # Set DTD parameters and trigger overlay
        if "dtd-run" == ctx.triggered_id:
            localDCXCache.clearDTD()
            localDCXCache.DTD_config.train_n_mixtures = train_n_mix
            localDCXCache.DTD_config.train_n_cells = train_n_cells
            localDCXCache.DTD_config.test_n_mixtures = test_n_mix
            localDCXCache.DTD_config.test_n_cells = test_n_cell
            localDCXCache.DTD_config.fRunOnAppl = fApplData
            localDCXCache.DTD_config.nIter = n_iter
            return True, True
        else:
            return False, no_update

    @app.callback(
        Output('dtd-exec-overlay', 'visible', allow_duplicate=True),
        Output('dtd-skeleton', 'visible', allow_duplicate=True),
        Output('dtd-res-loss', 'children', allow_duplicate=True),
        Output('dtd-res-corr', 'children', allow_duplicate=True),
        Output('dtd-res-mixtures', 'children', allow_duplicate=True),
        Output('dtd-res-markermap', 'children', allow_duplicate=True),
        Output('nav-adtd', 'disabled', allow_duplicate=True),
        Input('dtd-exec-overlay', 'visible'),
        State('dtd-par-check-ApplData', 'checked'),
        State('dtd-par-num-genes', 'value'),
        prevent_initial_call=True
    )
    def runDTD(exec_overlay_visible, runOnApplChecked, nGenes):
        # Run DTD simulation and update all result tabs
        if "dtd-exec-overlay" == ctx.triggered_id and exec_overlay_visible is True:
            _, localDCXCache.DTD_Y_train, localDCXCache.DTD_C_train = simulate_data(localDCXCache.DeconomixFile.Train,
                                                                                    localDCXCache.DTD_config.train_n_mixtures,
                                                                                    localDCXCache.DTD_config.train_n_cells,
                                                                                    n_genes=nGenes)
            _, localDCXCache.DTD_Y_test, localDCXCache.DTD_C_test = simulate_data(localDCXCache.DeconomixFile.Test,
                                                                                  localDCXCache.DTD_config.test_n_mixtures,
                                                                                  localDCXCache.DTD_config.test_n_cells)
            localDCXCache.DTD_Y_test = localDCXCache.DTD_Y_test.loc[localDCXCache.DTD_Y_train.index,:]
            localDCXCache.DTDmodel = DTD(localDCXCache.DeconomixFile.X_mat.loc[localDCXCache.DTD_Y_train.index,:],
                                         localDCXCache.DTD_Y_train,
                                         localDCXCache.DTD_C_train)
            localDCXCache.DTDmodel.run(localDCXCache.DTD_config.nIter)
            localDCXCache.DTD_C_train_est = calculate_estimated_composition(localDCXCache.DeconomixFile.X_mat.loc[localDCXCache.DTD_Y_train.index,:],
                                                                            localDCXCache.DTD_Y_train,
                                                                            localDCXCache.DTDmodel.gamma)
            localDCXCache.DTD_C_test_est = calculate_estimated_composition(localDCXCache.DeconomixFile.X_mat.loc[localDCXCache.DTD_Y_train.index,:],
                                                                           localDCXCache.DTD_Y_test,
                                                                           localDCXCache.DTDmodel.gamma)
            if localDCXCache.DeconomixFile.Application is not None and runOnApplChecked is True:
                localDCXCache.DTD_Y_appl = localDCXCache.DeconomixFile.Application.loc[localDCXCache.DTD_Y_train.index,:]
                localDCXCache.DTD_C_appl_est = calculate_estimated_composition(localDCXCache.DeconomixFile.X_mat.loc[localDCXCache.DTD_Y_train.index,:],
                                                                               localDCXCache.DTD_Y_appl,
                                                                               localDCXCache.DTDmodel.gamma)
            dtd_tab_loss = get_tab_dtd_loss(localDCXCache)
            dtd_tab_corr = get_tab_dtd_correlation(localDCXCache)
            dtd_tab_mix = get_tab_dtd_mixture(localDCXCache, runOnApplChecked)
            dtd_tab_marker = get_tab_dtd_markermap(localDCXCache)
            return False, False, dtd_tab_loss, dtd_tab_corr, dtd_tab_mix, dtd_tab_marker, False
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output("tab-dtd-scatter-train", "data"),
        Output("tab-dtd-scatter-test", "data"),
        Output("tab-dtd-train-corr", "children"),
        Output("tab-dtd-test-corr", "children"),
        Input("tab-dtd-correlation-combo", "value"),
        Input("tab-dtd-correlation-combo", "data")
    )
    def DTDselectCorrPlot(celltype, combobox_items):
        # Update correlation plots and correlation values for selected cell type
        idx = int(celltype[2:]) - 1
        label = combobox_items[idx]['label']
        data_train = get_correlation_dict(localDCXCache.DTD_C_train.loc[label],
                                          localDCXCache.DTD_C_train_est.loc[label],
                                          name='Training')
        data_test = get_correlation_dict(localDCXCache.DTD_C_test.loc[label],
                                         localDCXCache.DTD_C_test_est.loc[label],
                                         name='Testing')
        corr_train, _ = spearmanr(localDCXCache.DTD_C_train.loc[label],
                                  localDCXCache.DTD_C_train_est.loc[label])
        corr_val, _ = spearmanr(localDCXCache.DTD_C_test.loc[label],
                                localDCXCache.DTD_C_test_est.loc[label])
        return data_train, data_test, dmc.Text(f"Correlation: {np.round(corr_train, 4)}", ta="center"), dmc.Text(f"Correlation: {np.round(corr_val, 4)}", ta="center")

    @app.callback(
        Output("dtd-mix-mixture-combo", "data"),
        Input("dtd-mix-dataset-combo", "value")
    )
    def DTDselectMixtureDataSet(dataset):
        # Update mixture combobox based on selected dataset
        if dataset == "train":
            selected_data = localDCXCache.DTD_C_train_est
        elif dataset == "test":
            selected_data = localDCXCache.DTD_C_test_est
        elif dataset == "appl":
            selected_data = localDCXCache.DTD_C_appl_est
        mixture_data = fill_combo_mixtures(selected_data)
        return mixture_data

    @app.callback(
        Output("dtd-mix-pie-plot", "figure"),
        Input("dtd-mix-dataset-combo", "value"),
        Input("dtd-mix-mixture-combo", "value")
    )
    def UpdateDTDPiePlot(selectedDataset, selectedMixture):
        # Update pie chart for selected mixture
        if selectedDataset == "train":
            selected_data = localDCXCache.DTD_C_train_est
        elif selectedDataset == "test":
            selected_data = localDCXCache.DTD_C_test_est
        elif selectedDataset == "appl":
            selected_data = localDCXCache.DTD_C_appl_est
        mixture = selected_data.iloc[:, int(selectedMixture[1:]) - 1]
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

    @app.callback(
        Output("dtd-mix-dataset-download", "data"),
        Input("dtd-mix-dataset-button-download", "n_clicks"),
        State("dtd-mix-dataset-combo", "value")
    )
    def DTDDownloadEstimate(n_clicks, selectedDataset):
        # Download estimated composition for selected dataset
        if "dtd-mix-dataset-button-download" == ctx.triggered_id:
            if selectedDataset == "train":
                selected_data = localDCXCache.DTD_C_train_est
            elif selectedDataset == "test":
                selected_data = localDCXCache.DTD_C_test_est
            elif selectedDataset == "appl":
                selected_data = localDCXCache.DTD_C_appl_est
            return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_est.csv")
        else:
            return None

    @app.callback(
        Output("dtd-markermap-plot", "children"),
        Input("dtd-markermap-n-genes", "value")
    )
    def UpdateMarkermapPlot(nGenes):
        # Update markermap plot for selected number of genes
        markerplot = get_markermap_plot(localDCXCache, nGenes)
        return markerplot

    @app.callback(
        Output("dtd-gamma-download", "data"),
        Input("dtd-gamma-button-download", "n_clicks")
    )
    def DownloadGamma(n_clicks):
        # Download gamma vector
        if "dtd-gamma-button-download" == ctx.triggered_id:
            gamma = localDCXCache.DTDmodel.gamma.T.iloc[0]
            return dcc.send_data_frame(gamma.to_csv, f"gamma.csv")
        else:
            return None

    @app.callback(
        Input('dtd-skeleton', 'visible'),
        State("main-content", "children")
    )
    def storeCurrentDTDTab(skeletonVisible, currDTDTab):
        # Store current DTD tab in cache
        localDCXCache.DTDTab = currDTDTab

    @app.callback(
        Output("dtd-skeleton", "visible", allow_duplicate=True),
        Input("nav-dtd", "n_clicks"),
        prevent_initial_call=True
    )
    def show_dtd_skeleton_on_tab(n_clicks):
        # Only show skeleton if DTDmodel is None (i.e., no results yet), otherwise keep it hidden
        return localDCXCache.DTDmodel is None

def get_dtd_layout(applCheckEnabled, geneCount):
    from utils.global_cache import localDCXCache
    if geneCount == 0 and hasattr(localDCXCache, 'DeconomixFile') and localDCXCache.DeconomixFile is not None and hasattr(localDCXCache.DeconomixFile, 'X_mat') and localDCXCache.DeconomixFile.X_mat is not None:
        geneCount = localDCXCache.DeconomixFile.X_mat.shape[0]

    skeleton_visible = localDCXCache.DTDmodel is None

    # Prefill result components from cache if available
    if localDCXCache.DTDmodel is not None:
        res_loss = get_tab_dtd_loss(localDCXCache)
        res_corr = get_tab_dtd_correlation(localDCXCache)
        res_mixtures = get_tab_dtd_mixture(localDCXCache)
        res_markermap = get_tab_dtd_markermap(localDCXCache)
    else:
        res_loss = None
        res_corr = None
        res_mixtures = None
        res_markermap = None

    dtd_layout = dmc.Stack(
        [
            dmc.LoadingOverlay(
                visible=False,
                id="dtd-exec-overlay",
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
                                dmc.NumberInput(id="dtd-par-training-n-mix", label="Number of Mixtures", value=1000, min=1, allowDecimal=False, allowNegative=False),
                                dmc.NumberInput(id="dtd-par-training-n-cells-in-mix", label="Number of Cells in Mixtures", value=100, min=1, allowDecimal=False, allowNegative=False)
                            ],
                            legend="Training",
                            disabled=False,
                        ),
                        dmc.Fieldset(
                            children=[
                                dmc.NumberInput(id="dtd-par-test-n-mix", label="Number of Mixtures", value=1000, min=1, allowDecimal=False, allowNegative=False),
                                dmc.NumberInput(id="dtd-par-test-n-cells-in-mix", label="Number of Cells in Mixtures", value=100, min=1, allowDecimal=False, allowNegative=False)
                            ],
                            legend="Testing",
                            disabled=False,
                        ),
                        dmc.Fieldset(
                            children=[
                                dmc.Checkbox(id="dtd-par-check-ApplData", label="Run on Application Data", labelPosition='left', mb=10, disabled=not applCheckEnabled),
                                dmc.NumberInput(id="dtd-par-num-genes", label="Number of Genes", value=geneCount, max=geneCount, min=1, allowDecimal=False, allowNegative=False),
                                dmc.NumberInput(id="dtd-par-num-iter", label="Number of Iterations", value=1000, min=1, allowDecimal=False, allowNegative=False),
                                dmc.Button("Execute", id="dtd-run", fullWidth=True, mt=10)
                            ],
                            legend="DTD",
                            disabled=False,
                        )
                    ])

                ]
            ),

            # Results
            dmc.Skeleton(
                id="dtd-skeleton",
                visible=skeleton_visible,
                animate=False,
                children=[
                    dmc.Fieldset(
                        dmc.Tabs(
                            [
                                dmc.TabsList(
                                    [
                                        dmc.TabsTab("Loss", value="loss"),
                                        dmc.TabsTab("Correlation", value="correlation"),
                                        dmc.TabsTab("Markermap", value="markermap"),
                                        dmc.TabsTab("Estimated Mixtures", value="mixtures"),
                                    ]
                                ),
                                dmc.TabsPanel(value="loss",
                                              children=[
                                                  html.Div(id="dtd-res-loss", children=res_loss)
                                              ]),

                                dmc.TabsPanel(value="correlation",
                                              children=[
                                                  html.Div(id="dtd-res-corr", children=res_corr)
                                              ]),

                                dmc.TabsPanel(value="markermap",
                                              children=[
                                                  html.Div(id="dtd-res-markermap", children=res_markermap)
                                              ]),

                                dmc.TabsPanel(value="mixtures",
                                              children=[
                                                  html.Div(id="dtd-res-mixtures", children=res_mixtures)
                                              ])
                            ],
                            color='blue',
                            orientation='horizontal',
                            variant='default',
                            value="loss"
                        )
                    )
                ]
            )
        ]
    )

    return dtd_layout

def get_tab_dtd_loss(localDCXCache):
    loss_data = [{"epoch": str(i + 1), "corr": float(corr)} for i, corr in enumerate(localDCXCache.DTDmodel.mean_corr)]

    dtd_tab_loss = dmc.Stack(
        children=[
            dmc.LineChart(
                h=300,
                dataKey="epoch",
                data=loss_data,
                connectNulls=True,
                xAxisLabel='Epoch',
                yAxisLabel='Mean Correlation',
                series=[{"name": "corr", "color": "indigo.6"}],
                curveType="linear",
                withTooltip=False,
                withDots=False,
                mt=25
            )
        ]
    )

    return dtd_tab_loss

def get_tab_dtd_correlation(localDCXCache):

    combobox_items = [{'value': f"ct{i+1}", 'label': label} for i, label in enumerate(localDCXCache.DTD_C_train.index.unique())]

    data_train = get_correlation_dict(localDCXCache.DTD_C_train.loc[combobox_items[0]['label']],
                                      localDCXCache.DTD_C_train_est.loc[combobox_items[0]['label']],
                                      name='Training')
    data_test = get_correlation_dict(localDCXCache.DTD_C_test.loc[combobox_items[0]['label']],
                                      localDCXCache.DTD_C_test_est.loc[combobox_items[0]['label']],
                                      name='Testing')


    dtd_tab_correlation = dmc.Stack(
        children=[
            dmc.Select(
                label="Select Celltype for comparing Correlation",
                id="tab-dtd-correlation-combo",
                value='ct1',
                data=combobox_items,
                allowDeselect=False,
                mb=10
            ),
            dmc.Group(
                id='tab-dtd-correlation-scatter',
                w="100%",
                children=[
                    dmc.Fieldset(
                        w="48%",
                        children=[
                            dmc.Stack(
                                children=[
                                    dmc.Text("Training", ta="center"),
                                    html.Div(id="tab-dtd-train-corr"),
                                    dmc.ScatterChart(
                                        id="tab-dtd-scatter-train",
                                        h=300,
                                        data=data_train,
                                        dataKey={"x": "C_true", "y": "C_est"},
                                        xAxisLabel="True",
                                        yAxisLabel="Estimate",
                                        withTooltip=False
                                    )
                                ]
                            )
                        ]
                    ),

                    dmc.Fieldset(
                        w="48%",
                        children=[
                            dmc.Stack(
                                children=[
                                    dmc.Text("Testing", ta="center"),
                                    html.Div(id="tab-dtd-test-corr"),
                                    dmc.ScatterChart(
                                        id="tab-dtd-scatter-test",
                                        h=300,
                                        data=data_test,
                                        dataKey={"x": "C_true", "y": "C_est"},
                                        xAxisLabel="True",
                                        yAxisLabel="Estimate",
                                        withTooltip=False
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )

        ]

    )

    return dtd_tab_correlation

def get_correlation_dict(C_true, C_est, name="", color="blue.5"):
    data = [
        {
            "color": color,
            "name": name,
            "data": [{'C_true': C_true[i], 'C_est': C_est[i]} for i in range(0, len(C_true))]
        }]
    return data

def get_tab_dtd_markermap(localDCXCache):

    markermap_graph = get_markermap_plot(localDCXCache, 250)
    max_genes = len(localDCXCache.DTDmodel.gamma.T.iloc[0])
    layout_markermap = dmc.Stack(
        children=[
            dmc.Group(
                children=[
                    dmc.NumberInput(id="dtd-markermap-n-genes", label="Number of Genes", value=250, min=1, max=max_genes,
                                    allowDecimal=False, allowNegative=False),
                    dmc.Button("Download Gamma",
                               id="dtd-gamma-button-download",
                               mt=25
                               ),
                    dcc.Download(id="dtd-gamma-download")
                ]
            ),
            html.Div(id='dtd-markermap-plot', children=markermap_graph)
        ]
    )

    return layout_markermap

def get_markermap_plot(localDCXCache, nGenes):

    gamma = localDCXCache.DTDmodel.gamma.T.iloc[0]
    X = localDCXCache.DeconomixFile.X_mat

    genes = gamma.mul(X.var(axis=1).loc[gamma.index], axis=0).sort_values(ascending=False).index[0:nGenes]
    df = X.loc[genes]
    df = df.div(df.sum(axis=1), axis=0)

    markermap = dcc.Graph(figure=dashbio.Clustergram(
            data=df.T.values,
            row_labels=df.columns.to_list(),
            column_labels=genes.to_list(),
            color_map="viridis",
            display_ratio=[0.1, 0.2],
            hidden_labels='column'
        ),
        responsive=True,
        style = {"width": "100%", "height": "100%"}
    )

    return markermap

def get_tab_dtd_mixture(localDCXCache, fApplChecked=False):

    combobox_datasets_items = [
        {'value': 'train', 'label': "Training"},
        {'value': 'test', 'label': "Testing"}
    ]
    if localDCXCache.DTD_Y_appl is not None and fApplChecked is True:
        combobox_datasets_items.append({'value': 'appl', 'label': "Application"})

    combobox_mixture_items = fill_combo_mixtures(localDCXCache.DTD_C_train_est)

    mixture_data = [
        {'name': "Celltype 1", "value": 0.1, "color": "#1f77b4"},
        {'name': "Celltype 2", "value": 0.4, "color": "#ff7f0e"},
        {'name': "Celltype 3", "value": 0.2, "color": "#2ca02c"},
        {'name': "Celltype 4", "value": 0.1, "color": "#d62728"},
        {'name': "Celltype 5", "value": 0.15, "color": "#9467bd"},
        {'name': "Celltype 6", "value": 0.05, "color": "#8c564b"},
    ]

    pie_plot = go.Figure(
        data=[
            go.Pie(
                labels=[item['name'] for item in mixture_data],
                values=[item['value'] for item in mixture_data],
                marker=dict(colors=[item['color'] for item in mixture_data]),
                textinfo="label+value",
                showlegend = False,
                hoverinfo = "none",
                insidetextorientation="radial",
                textposition="outside"
            )
        ]
    )

    tab_dtd_mixture_layout = dmc.Stack(
        children=[
            dmc.Group(
                children=[
                    dmc.Select(label="Dataset",
                               id="dtd-mix-dataset-combo",
                               value='train',
                               data=combobox_datasets_items,
                               allowDeselect=False),
                    dmc.Select(label="Mixture",
                               id="dtd-mix-mixture-combo",
                               value='m1',
                               data=combobox_mixture_items,
                               allowDeselect=False),
                    dmc.Button("Download selected Dataset",
                                id="dtd-mix-dataset-button-download",
                               mt=25
                               ),
                    dcc.Download(id="dtd-mix-dataset-download")
                ]
            ),
            dmc.Group(
                children=[
                    dcc.Graph(id="dtd-mix-pie-plot",
                              figure=pie_plot,
                              responsive=True,
                              style = {"width": "100%", "height": "100%"})
                ]
            )
        ]
    )

    return tab_dtd_mixture_layout

def fill_combo_mixtures(dataset):
    nr_of_mixtures = len(dataset.columns)

    combobox_mixture_items = [
        {'value': f'm{i+1}', 'label': f"Mixture {i + 1}"} for i in range(0,nr_of_mixtures)
    ]

    return combobox_mixture_items