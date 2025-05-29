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
from utils.session_cache_manager import get_session_cache, session_manager

def get_layout(session_id, checkApplEnabled=True, n_genes=0):
    return get_dtd_layout(session_id, checkApplEnabled, n_genes)

def display_plugin(session_id):
    from utils.session_cache_manager import get_session_cache
    cache = get_session_cache(session_id)
    applCheckEnabled = False
    geneCount = 0
    dtd_tab = "loss"
    file_loaded = getattr(cache, "DeconomixFile", None) is not None
    dtd_executed = getattr(cache, "DTDmodel", None) is not None
    print(f"[DEBUG] Session-Wechsel: DTD_PAGE | session_id={session_id} | file_loaded={file_loaded} | dtd_executed={dtd_executed}")
    if file_loaded:
        applCheckEnabled = True
        geneCount = cache.DeconomixFile.X_mat.shape[0] if hasattr(cache.DeconomixFile, "X_mat") and cache.DeconomixFile.X_mat is not None else 0
        if dtd_executed:
            dtd_tab = cache.DTDTab if hasattr(cache, "DTDTab") else "loss"
    print(f"[DEBUG] DTD Tab aktiv: {dtd_tab}")
    layout = get_layout(session_id, applCheckEnabled, geneCount)
    # Tabs-Panel ggf. setzen
    if hasattr(layout, "props") and "children" in layout.props:
        for child in layout.props["children"]:
            if hasattr(child, "props") and child.props.get("id") == "dtd-tab-panel":
                child.props["value"] = dtd_tab
    return layout

def nav_disabled(session_id):
    disabled = True
    if session_id is not None:
        cache = get_session_cache(session_id)
        disabled = not cache or not cache.DeconomixFile
    return disabled

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

    @app.callback(
        Output('dtd-exec-overlay', 'visible', allow_duplicate=True),
        Output('dtd-skeleton', 'visible', allow_duplicate=True),
        State('dtd-par-training-n-mix', 'value'),
        State('dtd-par-training-n-cells-in-mix', 'value'),
        State('dtd-par-test-n-mix', 'value'),
        State('dtd-par-test-n-cells-in-mix', 'value'),
        State('dtd-par-check-ApplData', 'checked'),
        State('dtd-par-num-iter', 'value'),
        State('session-id', 'data'),
        Input('dtd-run', 'n_clicks'),
        prevent_initial_call=True
    )
    def prepareDTD(train_n_mix, train_n_cells, test_n_mix, test_n_cell, fApplData, n_iter, session_id, n_clicks):
        cache = get_session_cache(session_id)
        if "dtd-run" == ctx.triggered_id:
            cache.clearDTD()
            cache.DTD_config.train_n_mixtures = train_n_mix
            cache.DTD_config.train_n_cells = train_n_cells
            cache.DTD_config.test_n_mixtures = test_n_mix
            cache.DTD_config.test_n_cells = test_n_cell
            cache.DTD_config.fRunOnAppl = fApplData
            cache.DTD_config.nIter = n_iter
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
        Output('navbar', 'children', allow_duplicate=True),  # NEU: Navbar-Output
        Input('dtd-exec-overlay', 'visible'),
        State('session-id', 'data'),
        State('dtd-par-check-ApplData', 'checked'),
        State('dtd-par-num-genes', 'value'),
        prevent_initial_call=True
    )
    def runDTD(exec_overlay_visible, session_id, runOnApplChecked, nGenes):
        cache = get_session_cache(session_id)
        if "dtd-exec-overlay" == ctx.triggered_id and exec_overlay_visible is True:
            try:
                # Import jetzt lokal, um circular import zu vermeiden
                from DeconomixApp import get_nav_links
                _, cache.DTD_Y_train, cache.DTD_C_train = simulate_data(cache.DeconomixFile.Train,
                                                                        cache.DTD_config.train_n_mixtures,
                                                                        cache.DTD_config.train_n_cells,
                                                                        n_genes=nGenes)
                _, cache.DTD_Y_test, cache.DTD_C_test = simulate_data(cache.DeconomixFile.Test,
                                                                      cache.DTD_config.test_n_mixtures,
                                                                      cache.DTD_config.test_n_cells)
                cache.DTD_Y_test = cache.DTD_Y_test.loc[cache.DTD_Y_train.index, :]
                cache.DTDmodel = DTD(cache.DeconomixFile.X_mat.loc[cache.DTD_Y_train.index, :],
                                     cache.DTD_Y_train,
                                     cache.DTD_C_train)
                cache.DTDmodel.run(cache.DTD_config.nIter)
                cache.DTD_C_train_est = calculate_estimated_composition(cache.DeconomixFile.X_mat.loc[cache.DTD_Y_train.index, :],
                                                                        cache.DTD_Y_train,
                                                                        cache.DTDmodel.gamma)
                cache.DTD_C_test_est = calculate_estimated_composition(cache.DeconomixFile.X_mat.loc[cache.DTD_Y_train.index, :],
                                                                       cache.DTD_Y_test,
                                                                       cache.DTDmodel.gamma)
                if cache.DeconomixFile.Application is not None and runOnApplChecked is True:
                    cache.DTD_Y_appl = cache.DeconomixFile.Application.loc[cache.DTD_Y_train.index, :]
                    cache.DTD_C_appl_est = calculate_estimated_composition(cache.DeconomixFile.X_mat.loc[cache.DTD_Y_train.index, :],
                                                                           cache.DTD_Y_appl,
                                                                           cache.DTDmodel.gamma)
                dtd_tab_loss = get_tab_dtd_loss(cache)
                dtd_tab_corr = get_tab_dtd_correlation(cache)
                dtd_tab_mix = get_tab_dtd_mixture(cache, runOnApplChecked)
                dtd_tab_marker = get_tab_dtd_markermap(cache)
                # Persistiere Session nach erfolgreichem DTD-Lauf
                session_manager.save_session(session_id)
                nav_links = get_nav_links(session_id)  # Navbar aktualisieren
                return False, False, dtd_tab_loss, dtd_tab_corr, dtd_tab_mix, dtd_tab_marker, nav_links
            except Exception as e:
                return False, False, f"Error: {e}", None, None, None, no_update
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output("tab-dtd-scatter-train", "data"),
        Output("tab-dtd-scatter-test", "data"),
        Output("tab-dtd-train-corr", "children"),
        Output("tab-dtd-test-corr", "children"),
        State('session-id', 'data'),
        Input("tab-dtd-correlation-combo", "value"),
        Input("tab-dtd-correlation-combo", "data")
    )
    def DTDselectCorrPlot(session_id, celltype, combobox_items):
        cache = get_session_cache(session_id)
        idx = int(celltype[2:]) - 1
        label = combobox_items[idx]['label']
        data_train = get_correlation_dict(cache.DTD_C_train.loc[label],
                                          cache.DTD_C_train_est.loc[label],
                                          name='Training')
        data_test = get_correlation_dict(cache.DTD_C_test.loc[label],
                                         cache.DTD_C_test_est.loc[label],
                                         name='Testing')
        corr_train, _ = spearmanr(cache.DTD_C_train.loc[label],
                                  cache.DTD_C_train_est.loc[label])
        corr_val, _ = spearmanr(cache.DTD_C_test.loc[label],
                                cache.DTD_C_test_est.loc[label])
        return data_train, data_test, dmc.Text(f"Correlation: {np.round(corr_train, 4)}", ta="center"), dmc.Text(f"Correlation: {np.round(corr_val, 4)}", ta="center")

    @app.callback(
        Output("dtd-mix-mixture-combo", "data"),
        State('session-id', 'data'),
        Input("dtd-mix-dataset-combo", "value")
    )
    def DTDselectMixtureDataSet(session_id, dataset):
        cache = get_session_cache(session_id)
        if dataset == "train":
            selected_data = cache.DTD_C_train_est
        elif dataset == "test":
            selected_data = cache.DTD_C_test_est
        elif dataset == "appl":
            selected_data = cache.DTD_C_appl_est
        mixture_data = fill_combo_mixtures(selected_data)
        return mixture_data

    @app.callback(
        Output("dtd-mix-pie-plot", "figure"),
        State('session-id', 'data'),
        Input("dtd-mix-dataset-combo", "value"),
        Input("dtd-mix-mixture-combo", "value")
    )
    def UpdateDTDPiePlot(session_id, selectedDataset, selectedMixture):
        cache = get_session_cache(session_id)
        if selectedDataset == "train":
            selected_data = cache.DTD_C_train_est
        elif selectedDataset == "test":
            selected_data = cache.DTD_C_test_est
        elif selectedDataset == "appl":
            selected_data = cache.DTD_C_appl_est
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
        State('session-id', 'data'),
        Input("dtd-mix-dataset-button-download", "n_clicks"),
        State("dtd-mix-dataset-combo", "value")
    )
    def DTDDownloadEstimate(session_id, n_clicks, selectedDataset):
        cache = get_session_cache(session_id)
        if "dtd-mix-dataset-button-download" == ctx.triggered_id:
            if selectedDataset == "train":
                selected_data = cache.DTD_C_train_est
            elif selectedDataset == "test":
                selected_data = cache.DTD_C_test_est
            elif selectedDataset == "appl":
                selected_data = cache.DTD_C_appl_est
            return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_est.csv")
        else:
            return None

    @app.callback(
        Output("dtd-markermap-plot", "children"),
        State('session-id', 'data'),
        Input("dtd-markermap-n-genes", "value")
    )
    def UpdateMarkermapPlot(session_id, nGenes):
        cache = get_session_cache(session_id)
        markerplot = get_markermap_plot(cache, nGenes)
        return markerplot

    @app.callback(
        Output("dtd-gamma-download", "data"),
        State('session-id', 'data'),
        Input("dtd-gamma-button-download", "n_clicks")
    )
    def DownloadGamma(session_id, n_clicks):
        cache = get_session_cache(session_id)
        if "dtd-gamma-button-download" == ctx.triggered_id:
            gamma = cache.DTDmodel.gamma.T.iloc[0]
            return dcc.send_data_frame(gamma.to_csv, f"gamma.csv")
        else:
            return None

    @app.callback(
        State('session-id', 'data'),
        Input('dtd-skeleton', 'visible'),
        State("main-content", "children")
    )
    def storeCurrentDTDTab(session_id, skeletonVisible, currDTDTab):
        cache = get_session_cache(session_id)
        cache.DTDTab = currDTDTab

    @app.callback(
        Output("dtd-skeleton", "visible", allow_duplicate=True),
        State('session-id', 'data'),
        Input("nav-dtd_page", "n_clicks"),
        prevent_initial_call=True
    )
    def show_dtd_skeleton_on_tab(session_id, n_clicks):
        cache = get_session_cache(session_id)
        return cache.DTDmodel is None

def get_dtd_layout(session_id, applCheckEnabled, geneCount):
    cache = get_session_cache(session_id)
    if geneCount == 0 and hasattr(cache, 'DeconomixFile') and cache.DeconomixFile is not None and hasattr(cache.DeconomixFile, 'X_mat') and cache.DeconomixFile.X_mat is not None:
        geneCount = cache.DeconomixFile.X_mat.shape[0]

    skeleton_visible = cache.DTDmodel is None

    if cache.DTDmodel is not None:
        res_loss = get_tab_dtd_loss(cache)
        res_corr = get_tab_dtd_correlation(cache)
        res_mixtures = get_tab_dtd_mixture(cache)
        res_markermap = get_tab_dtd_markermap(cache)
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

def get_tab_dtd_loss(cache):
    loss_data = [{"epoch": str(i + 1), "corr": float(corr)} for i, corr in enumerate(cache.DTDmodel.mean_corr)]

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

def get_tab_dtd_correlation(cache):

    combobox_items = [{'value': f"ct{i+1}", 'label': label} for i, label in enumerate(cache.DTD_C_train.index.unique())]

    data_train = get_correlation_dict(cache.DTD_C_train.loc[combobox_items[0]['label']],
                                      cache.DTD_C_train_est.loc[combobox_items[0]['label']],
                                      name='Training')
    data_test = get_correlation_dict(cache.DTD_C_test.loc[combobox_items[0]['label']],
                                      cache.DTD_C_test_est.loc[combobox_items[0]['label']],
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

def get_tab_dtd_markermap(cache):

    markermap_graph = get_markermap_plot(cache, 250)
    max_genes = len(cache.DTDmodel.gamma.T.iloc[0])
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

def get_markermap_plot(cache, nGenes):

    gamma = cache.DTDmodel.gamma.T.iloc[0]
    X = cache.DeconomixFile.X_mat

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

def get_tab_dtd_mixture(cache, fApplChecked=False):

    combobox_datasets_items = [
        {'value': 'train', 'label': "Training"},
        {'value': 'test', 'label': "Testing"}
    ]
    if cache.DTD_Y_appl is not None and fApplChecked is True:
        combobox_datasets_items.append({'value': 'appl', 'label': "Application"})

    combobox_mixture_items = fill_combo_mixtures(cache.DTD_C_train_est)

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