####################################################
#
# ADTD page for Deconomix GUI application
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
from utils.session_cache_manager import get_session_cache, session_manager
from dash.exceptions import PreventUpdate

def get_layout(session_id, checkApplEnabled=True):
    cache = get_session_cache(session_id)
    return get_adtd_layout(session_id, checkApplEnabled)

def display_plugin(session_id):
    from utils.session_cache_manager import get_session_cache
    cache = get_session_cache(session_id)
    applCheckEnabled = getattr(cache, "DTDmodel", None) is not None
    adtd_tab = "mixtures"
    adtd_executed = getattr(cache, "ADTDmodel", None) is not None

    layout = get_layout(session_id, applCheckEnabled)

    if hasattr(cache, "ADTD_results_cache") and cache.ADTD_results_cache:
        first_key = next(iter(cache.ADTD_results_cache.keys()))
        dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic = first_key
        cache.ADTD_config.Dataset = dataset
        layout["adtd-dataset-combo"].value = dataset
        cache.ADTD_config.lambda1 = lambda1
        layout["adtd-par-lambda1"].value = lambda1
        cache.ADTD_config.lambda2 = lambda2
        layout["adtd-par-lambda2"].value = lambda2
        cache.ADTD_config.nIter = nIter
        layout["adtd-par-iterations"].value = nIter
        cache.ADTD_config.Cstatic = Cstatic
        layout["adtd-par-check-Cstatic"].checked = Cstatic
        cache.ADTD_config.Deltastatic = Deltastatic
        layout["adtd-par-check-Deltastatic"].checked = Deltastatic
        layout['adtd-res-mixtures'].children = cache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)['tab_mixture'] 
        layout['adtd-res-gr'].children = cache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)['tab_gr']
        layout['adtd-tab-gr'].disabled = Deltastatic
        layout["adtd-skeleton"].visible = False
        layout["adtd-tab-panel"].value = "mixtures"
    
    print(f"[DEBUG] Session-Wechsel: ADTD_PAGE | session_id={session_id} | dtd_executed={applCheckEnabled} | adtd_executed={adtd_executed}")
    print(f"[DEBUG] ADTD Tab aktiv: {adtd_tab}")
    
    return layout

def nav_disabled(session_id):
    disabled = True
    if session_id is not None:
        cache = get_session_cache(session_id)
        disabled = getattr(cache, "DeconomixFile", None) is None or getattr(cache, "DTDmodel", None) is None
    return disabled

def register_callbacks(app):
    from dash import Output, Input, State, no_update, ctx, ALL, callback_context
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
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def updateLambda1Input(CstaticChecked, session_id):
        cache = get_session_cache(session_id)
        if CstaticChecked: return True
        else: return False

    # Callback for updating lambda2 input field
    @app.callback(
        Output('adtd-par-lambda2', 'disabled'),
        Input('adtd-par-check-Deltastatic', 'checked'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def updateLambda2Input(DeltastaticChecked, session_id):
        cache = get_session_cache(session_id)
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
        State('session-id', 'data'),
        running=[(Output("adtd-skeleton", "visible"), True, False),
                 (Output("adtd-exec-overlay", "visible"), True, False)],
        prevent_initial_call=True
    )
    def runADTDHPS(n_clicks, lambda_min, lambda_max, npoints, dataset, gr_disabled, mix_disabled, session_id):
        cache = get_session_cache(session_id)
        cache.ADTD_config.lambda2min = lambda_min
        cache.ADTD_config.lambda2max = lambda_max
        cache.ADTD_config.nPoints = npoints
        cache.ADTD_config.Dataset = dataset
        if dataset == 'train':
            Y_hps = cache.DTD_Y_train
        elif dataset == 'appl':
            Y_hps = cache.DeconomixFile.Application.loc[cache.DTDmodel.gamma.index,:]
        else:
            Y_hps = cache.DTD_Y_test
        cache.ADTD_HPS_model = deconomix.methods.HPS(cache.DeconomixFile.X_mat.loc[Y_hps.index,:],
                                                     Y_hps,
                                                     cache.DTDmodel.gamma,
                                                     lambda_min=lambda_min,
                                                     lambda_max=lambda_max,
                                                     n_points=npoints)
        cache.ADTD_HPS_model.run()
        fig = getHPSPlot(cache)
        optimalLambda2 = cache.ADTD_HPS_model.lambda_max_gradient
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
        State('session-id', 'data'),
        running=[(Output("adtd-skeleton", "visible"), True, False),
                 (Output("adtd-exec-overlay", "visible"), True, False)],
        prevent_initial_call=True
    )
    def runADTD(n_clicks, Cstatic, Deltastatic, lambda1, lambda2, dataset, nIter, hps_disabled, gr_disabled, mix_disabled, session_id):
        cache = get_session_cache(session_id)
        cached = cache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)
        if cached is not None:
            return cached['tab_mixture'], cached['tab_gr'], 'mixtures', hps_disabled, Deltastatic, False
        previousDataset = cache.ADTD_config.Dataset
        cache.ADTD_config.Cstatic = Cstatic
        cache.ADTD_config.Deltastatic = Deltastatic
        cache.ADTD_config.lambda1 = lambda1
        cache.ADTD_config.lambda2 = lambda2
        cache.ADTD_config.Dataset = dataset
        cache.ADTD_config.nIter = nIter
        if dataset == 'train':
            Y_adtd = cache.DTD_Y_train
        elif dataset == 'appl':
            Y_adtd = cache.DeconomixFile.Application.loc[cache.DTDmodel.gamma.index,:]
        else:
            Y_adtd = cache.DTD_Y_test
        if not hps_disabled \
                and cache.ADTD_config.Dataset == previousDataset \
                and cache.ADTD_HPS_model is not None \
                and lambda2 == cache.ADTD_HPS_model.lambda_max_gradient:
            gamma = 1 / Y_adtd.shape[1] * np.ones(Y_adtd.shape[0]) / (Y_adtd.mean(axis=1))**2
        else:
            gamma = cache.DTDmodel.gamma
        cache.ADTDmodel = deconomix.methods.ADTD(cache.DeconomixFile.X_mat.loc[Y_adtd.index,:],
                                                 Y_adtd,
                                                 gamma,
                                                 lambda1,
                                                 lambda2,
                                                 nIter,
                                                 C_static=Cstatic,
                                                 Delta_static=Deltastatic)
        cache.ADTDmodel.run()
        tab_mixture = get_tab_adtd_mixture(cache, cache.ADTDmodel.C_est, cache.ADTDmodel.c_est)
        if not Deltastatic:
            tab_gr = get_tab_adtd_geneRegulation(cache, dataset)
        else:
            tab_gr = html.Div("Something went wrong, you shouldn't see this!")
        cache.set_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic, {
            'tab_mixture': tab_mixture,
            'tab_gr': tab_gr,
            'C_est': cache.ADTDmodel.C_est.copy(),
            'c_est': cache.ADTDmodel.c_est.copy()
        })
        # Save session state
        session_manager.save_session(session_id)
        return tab_mixture, tab_gr, 'mixtures', hps_disabled, Deltastatic, False

    # Callback for Skeleton handling when dataset changes
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
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def restoreADTDResultsOnDatasetChange(dataset, Cstatic, Deltastatic, lambda1, lambda2, nIter, hps_disabled, gr_disabled, mix_disabled, session_id):
        cache = get_session_cache(session_id)
        cache.ADTD_config.Dataset = dataset
        cached = cache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)
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
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def UpdateADTDPiePlot(selectedMixture, dataset, Cstatic, Deltastatic, lambda1, lambda2, nIter, session_id):
        cache = get_session_cache(session_id)
        cached = cache.get_adtd_result(dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic)
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
        return go.Figure()

    # Callback for downloading ADTD estimated composition
    @app.callback(
        Output("adtd-mix-dataset-download", "data"),
        Input("adtd-mix-dataset-button-download", "n_clicks"),
        State("adtd-dataset-combo", "value"),
        State('session-id', 'data')
    )
    def ADTDDownloadEstimate(n_clicks, selectedDataset, session_id):
        cache = get_session_cache(session_id)
        if "adtd-mix-dataset-button-download" == ctx.triggered_id:
            selected_data = cache.ADTDmodel.C_est.copy()
            selected_data.loc['hidden'] = cache.ADTDmodel.c_est.iloc[0]
            return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_ADTD_est.csv")
        else:
            return None

    # Callback for updating gene regulation plot
    @app.callback(
        Output("adtd-gr-selected-genes", "error"),
        Output("adtd-gr-plot", "figure"),
        Input("adtd-gr-selected-genes", "value"),
        State("adtd-gr-plot", "figure"),
        State('session-id', 'data')
    )
    def UpdateGeneRegulationPlot(genes, curr_fig, session_id):
        cache = get_session_cache(session_id)
        error = ""
        fig = curr_fig
        if len(genes) < 1:
            error = "Select at least one gene!"
        else:
            fig = get_gr_plot(cache, sorted(genes, key=str.lower))
        return error, fig

    # Callback for downloading gene regulation results
    @app.callback(
        Output("adtd-gr-download", "data"),
        Input("adtd-gr-button-download", "n_clicks"),
        State("adtd-dataset-combo", "value"),
        State('session-id', 'data')
    )
    def ADTDDownloadGR(n_clicks, selectedDataset, session_id):
        cache = get_session_cache(session_id)
        if "adtd-gr-button-download" == ctx.triggered_id:
            selected_data = cache.ADTDmodel.Delta_est
            return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_Delta.csv")
        else:
            return None

    # Callback for storing current ADTD tab in cache
    @app.callback(
        Input('adtd-skeleton', 'visible'),
        State("main-content", "children"),
        State('session-id', 'data')
    )
    def storeCurrentADTDTab(skeletonVisible, currADTDTab, session_id):
        cache = get_session_cache(session_id)
        cache.ADTDTab = currADTDTab

    
    @app.callback(
        Output('adtd-par-lambda1', 'value', allow_duplicate=True),
        Output('adtd-par-lambda2', 'value', allow_duplicate=True),
        Output('adtd-par-iterations', 'value', allow_duplicate=True),
        Output('adtd-dataset-combo', 'value', allow_duplicate=True),
        Output('adtd-par-check-Cstatic', 'checked', allow_duplicate=True),
        Output('adtd-par-check-Deltastatic', 'checked', allow_duplicate=True),
        Output('adtd-tab-panel', 'value', allow_duplicate=True),
        Output('adtd-previous-runs-modal', 'opened', allow_duplicate=True),
        Input({'type': 'adtd-show-previous', 'index': ALL}, 'n_clicks'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def show_previous_run(n_clicks_list, session_id):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate
        btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
        import json
        btn_idx = json.loads(btn_id)['index']
        cache = get_session_cache(session_id)
        key = list(cache.ADTD_results_cache.keys())[btn_idx]
        dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic = key
        return lambda1, lambda2, nIter, dataset, Cstatic, Deltastatic, 'mixtures', False

    @app.callback(
        Output("adtd-previous-runs-modal", "opened", allow_duplicate=True),
        Input("adtd-show-previous-modal-btn", "n_clicks"),
        Input("adtd-previous-runs-modal", "close"),
        State("adtd-previous-runs-modal", "opened"),
        prevent_initial_call=True
    )
    def toggle_previous_runs_modal(open_btn, close_evt, opened):
        ctx_id = ctx.triggered_id
        if ctx_id == "adtd-show-previous-modal-btn":
            return True
        elif ctx_id == "adtd-previous-runs-modal":
            return False
        return opened

    @app.callback(
        Output("adtd-previous-runs-table", "children", allow_duplicate=True),
        Input("adtd-previous-runs-modal", "opened"),
        State('session-id', 'data'),
        State('adtd-dataset-combo', 'value'),
        State('adtd-par-lambda1', 'value'),
        State('adtd-par-lambda2', 'value'),
        State('adtd-par-iterations', 'value'),
        State('adtd-par-check-Cstatic', 'checked'),
        State('adtd-par-check-Deltastatic', 'checked'),
        prevent_initial_call=True
    )
    def fill_previous_runs_table(opened, session_id, current_dataset, current_lambda1, current_lambda2, current_nIter, current_Cstatic, current_Deltastatic):
        if not opened:
            return no_update
        cache = get_session_cache(session_id)
        runs = getattr(cache, "ADTD_results_cache", None)
        if not runs or len(runs) == 0:
            return dmc.Alert("No previous runs found.", color="gray", variant="light")
        # Table header
        header = dmc.Group([
            dmc.Text("Dataset", fw=700, w=120),
            dmc.Text("Lambda1", fw=700, w=100),
            dmc.Text("Lambda2", fw=700, w=100),
            dmc.Text("Iterations", fw=700, w=100),
            dmc.Text("C static", fw=700, w=80),
            dmc.Text("Delta static", fw=700, w=100),
            dmc.Text("", w=120)
        ], gap="xs", mb=8)
        # Table rows
        rows = []
        # Mapping for dataset display names
        dataset_display = {"train": "Training", "test": "Testing", "appl": "Application"}
        for idx, key in enumerate(runs.keys()):
            dataset, lambda1, lambda2, nIter, Cstatic, Deltastatic = key
            is_active = (
                str(dataset) == str(current_dataset)
                and str(lambda1) == str(current_lambda1)
                and str(lambda2) == str(current_lambda2)
                and str(nIter) == str(current_nIter)
                and str(Cstatic) == str(current_Cstatic)
                and str(Deltastatic) == str(current_Deltastatic)
            )
            display_dataset = dataset_display.get(str(dataset), str(dataset))
            paper_style = {
                "backgroundColor": "#e6f7ff",
                "border": "2px solid #228be6",
                "transition": "background 0.2s, border 0.2s"
            } if is_active else {"backgroundColor": "#f8fafc", "transition": "background 0.2s, border 0.2s"}
            rows.append(
                dmc.Paper(
                    dmc.Group([
                        dmc.Text(display_dataset, w=120),
                        dmc.Text(str(lambda1), w=100),
                        dmc.Text(str(lambda2), w=100),
                        dmc.Text(str(nIter), w=100),
                        dmc.Text(str(Cstatic), w=80),
                        dmc.Text(str(Deltastatic), w=100),
                        dmc.Button("Restore", id={"type": "adtd-show-previous", "index": idx}, size="xs", color="blue", variant="outline", ml=8, style={"verticalAlign": "middle"})
                    ], gap="xs", align="center", mb=0, style={"width": "100%"}),
                    withBorder=True,
                    shadow="xs",
                    radius="md",
                    p="xs",
                    mb=6,
                    style=paper_style
                )
            )
        return dmc.Stack([header] + rows, gap=0)

def get_adtd_layout(session_id, applCheckEnabled):
    cache = get_session_cache(session_id)
    if getattr(cache, 'DeconomixFile', None) is None:
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
    if cache.DeconomixFile.Application is not None:
        combobox_datasets_items.append({'value': 'appl', 'label': "Application"})

    adtd_layout = dmc.Stack(
        [
            html.Div(id="adtd-notify-container"),
            dmc.LoadingOverlay(
                visible=False,
                id="adtd-exec-overlay",
                loaderProps={"type": "oval", "color": "red", "size": "lg"},
                overlayProps={"radius": "sm", "blur": 2}
            ),
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
                                dmc.Button("Show Previous Runs", id="adtd-show-previous-modal-btn", fullWidth=True, mt=10, color="gray"),
                            ],
                            legend="ADTD",
                            disabled=False,
                        )
                    ])
                ]
            ),
            # Modal for previous runs
            dmc.Modal(
                id="adtd-previous-runs-modal",
                title="Previous ADTD Runs",
                centered=True,
                size="xl",
                styles={"modal": {"minWidth": 800, "maxWidth": 1100}},
                children=[
                    html.Div(id="adtd-previous-runs-table")
                ],
                opened=False,
                closeOnClickOutside=True,
            ),
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


def getHPSPlot(cache):
    Losses = cache.ADTD_HPS_model.Losses

    avgLoss = Losses.mean(axis=0)
    stdLoss = Losses.std(axis=0)
    x_values = Losses.columns.to_list()

    fig = go.Figure()

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

def get_tab_adtd_mixture(cache, dataset, hidden):

    complete_dataset = dataset.copy()
    complete_dataset.loc['hidden'] = hidden.iloc[0]

    combobox_mixtures_items = fill_combo_mixtures(complete_dataset)

    selected_data = cache.ADTDmodel.C_est.copy()
    selected_data.loc['hidden'] = cache.ADTDmodel.c_est.iloc[0]

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
                               data=combobox_mixtures_items,
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

def get_tab_adtd_geneRegulation(cache, dataset):

    geneList = list(cache.DeconomixFile.X_mat.sort_index().index.unique())

    initially_selected = list(cache.ADTDmodel.Delta_est.abs().max(axis=1).sort_values(ascending=False).index.drop_duplicates())[:10]

    gr_plot_init = get_gr_plot(cache, initially_selected)

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


def get_gr_plot(cache, selected_genes):

    cell_types = list(cache.ADTDmodel.C_est.index.unique())
    delta = cache.ADTDmodel.Delta_est

    norm = CenteredNorm(1)
    normalized_delta = pd.DataFrame(norm(delta),
                                    index=delta.index,
                                    columns=delta.columns).loc[selected_genes, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=normalized_delta,
            x=cell_types,
            y=selected_genes,
            colorscale="PuOr",
            colorbar=dict(),
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
