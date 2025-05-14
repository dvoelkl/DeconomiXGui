####################################################
#
# Main Entry point for DeconomiX GUI application
#
####################################################


######################### IMPORTS #########################

import dash_mantine_components as dmc
import deconomix.methods
from dash import Dash, Input, Output, State, callback, _dash_renderer, dcc, html, ctx, DiskcacheManager, set_props, \
    no_update
from pages.Uploading import get_upload_layout, get_file_properties_layout, get_distribution_plot
from pages.DTD_page import get_dtd_layout, get_tab_dtd_loss, get_tab_dtd_correlation, \
    get_tab_dtd_mixture, get_correlation_dict, fill_combo_mixtures, get_tab_dtd_markermap, get_markermap_plot
from pages.ADTD_page import get_adtd_layout, getHPSPlot, get_tab_adtd_mixture, get_tab_adtd_geneRegulation, get_gr_plot
import plotly.graph_objects as go

import matplotlib
import numpy as np

import base64
import io
import pandas as pd


# Imports for Deconomix
from utils.DeconomixFile import DeconomixFile
from utils.DTD_config import DTDConfig
from utils.ADTD_config import ADTDConfig
from utils.DeconomixCache import DCXCache
from scipy.stats import spearmanr

# Import from Deconomix
from deconomix.utils import simulate_data, calculate_estimated_composition
from deconomix.methods import DTD

#########################  #########################

######################### GLOBALS #########################
localDCXCache = DCXCache(None, DTDConfig(), ADTD_config=ADTDConfig())

#########################  #########################


######################### GUI #########################

_dash_renderer._set_react_version("18.2.0")

app = Dash(external_stylesheets=dmc.styles.ALL, suppress_callback_exceptions=True)

logo_filename = "DeconomiX_Logo.png"

### APP SHELL LAYOUT ###
layout = dmc.AppShell(
    [
        dmc.AppShellHeader(
            dmc.Group(
                [
                    dmc.Burger(
                        id="mobile-burger",
                        size="sm",
                        hiddenFrom="sm",
                        opened=False,
                    ),
                    dmc.Burger(
                        id="desktop-burger",
                        size="sm",
                        visibleFrom="sm",
                        opened=True,
                    ),
                    dmc.Image(src=app.get_asset_url(logo_filename), h=40),
                    dmc.Title("DeconomiX", c="red"),
                ],
                h="100%",
                px="md",
            )
        ),
        dmc.AppShellNavbar(
            id="navbar",
            children=[
                dmc.NavLink(label="Load", description="Load a DeconomiX file", id="nav-load"),
                dmc.NavLink(label="DTD", description="Run DTD", id="nav-dtd", disabled=True),
                dmc.NavLink(label="ADTD", description="Run ADTD", id="nav-adtd", disabled=True),
            ],
            p="md",
        ),
        dmc.AppShellMain(id="main-content",
                         children=get_upload_layout()),
    ],
    header={"height": 60},
    navbar={
        "width": 300,
        "breakpoint": "sm",
        "collapsed": {"mobile": True, "desktop": False},
    },
    padding="md",
    id="appshell",
)

app.layout = dmc.MantineProvider(layout)


### Callbacks Navigation ###
@callback(
    Output("appshell", "navbar"),
    Input("mobile-burger", "opened"),
    Input("desktop-burger", "opened"),
    State("appshell", "navbar"),
)
def toggle_navbar(mobile_opened, desktop_opened, navbar):
    navbar["collapsed"] = {
        "mobile": not mobile_opened,
        "desktop": not desktop_opened,
    }
    return navbar


@callback(
    Output("main-content", "children", allow_duplicate=True),
    Input("nav-load", "n_clicks"),
    prevent_initial_call=True,
)
def display_load(n_clicks):
    return get_upload_layout()


@callback(
    Output("main-content", "children", allow_duplicate=True),
    Input("nav-dtd", "n_clicks"),
    prevent_initial_call=True,
)
def display_dtd(n_clicks):
    global localDCXCache

    if localDCXCache.DTDTab is not None:
        return localDCXCache.DTDTab

    if localDCXCache.DeconomixFile.Application is not None:
        checkApplEnabled = True
    else:
        checkApplEnabled = False

    localDCXCache.DTDTab = get_dtd_layout(checkApplEnabled, localDCXCache.DeconomixFile.X_mat.shape[0])
    return localDCXCache.DTDTab


@callback(
    Output("main-content", "children", allow_duplicate=True),
    Input("nav-adtd", "n_clicks"),
    prevent_initial_call=True,
)
def display_adtd(n_clicks):
    global localDCXCache

    if localDCXCache.ADTDTab is not None:
        return localDCXCache.ADTDTab

    if localDCXCache.DeconomixFile.Application is not None:
        checkApplEnabled = True
    else:
        checkApplEnabled = False

    localDCXCache.ADTDTab = get_adtd_layout(checkApplEnabled, localDCXCache)
    return localDCXCache.ADTDTab


### Callbacks Load ###
@callback(
    Output('upload-file-properties', 'children'),
    Output('nav-dtd', 'disabled'),
    Output('nav-adtd', 'disabled', allow_duplicate=True),
    Output('upload-file-not-supported-alert', 'opened'),
    Input('upload-file', 'contents'),
    Input('upload-file', 'filename'),
    Input('upload-file', 'last_modified'),
    State('nav-dtd', 'disabled'),
    State('nav-adtd', 'disabled'),
    prevent_initial_call='initial_duplicate')
def uploadDeconomixFile(content, filename, last_modified, dtd_was_disabled, adtd_was_disabled):
    global localDCXCache
    # Check if content can be parsed as DeconomiX File

    showAlert = False

    # Enable/Disable Other tabs
    if filename is None and localDCXCache.DeconomixFile is None:
        disabled_dtd = True
        disabled_adtd = True
        file_properties_layout = "No file loaded!"
        localDCXCache.DeconomixFile = None
    else:
        disabled_dtd = False
        disabled_adtd = True

        if filename is not None:
            _, content = content.split(',')
            content = io.BytesIO(base64.b64decode(content))

            try:
                localDCXCache.DeconomixFile = DeconomixFile.unbinarize(content)
                localDCXCache.DeconomixFile.filename = filename
                file_properties_layout = get_file_properties_layout(filename, last_modified, localDCXCache.DeconomixFile)

                # Remove stored DTD and ADTD Tab, as file has changed
                localDCXCache.clearAll()

            except:
                showAlert = True
                disabled_dtd = True
                disabled_adtd = True
                file_properties_layout = "No file loaded!"
                localDCXCache.DeconomixFile = None


        else:
            file_properties_layout = get_file_properties_layout(localDCXCache.DeconomixFile.filename, "",
                                                                localDCXCache.DeconomixFile)

        if localDCXCache.DTDmodel is not None:
            disabled_adtd = False

    return file_properties_layout, disabled_dtd, disabled_adtd, showAlert


@callback(
    Output("import-anndata-modal", "opened"),
    Input("anndata-trigger-import", "n_clicks"),
    Input("anndata-cancel-btn", "n_clicks"),
    State("import-anndata-modal", "opened"),
    prevent_initial_call=True,
)
def toggle_modal(open_click, cancel_click, modalOpen):
    return ~modalOpen

@callback(
    [
        Output("anndata-indicator-x", "style"),
        Output("anndata-indicator-train", "style"),
        Output("anndata-indicator-test", "style"),
        Output("anndata-indicator-app", "style"),
        Output("anndata-import-btn", "disabled"),
    ],
    [
        Input("anndata-upload-x", "contents"),
        Input("anndata-upload-train", "contents"),
        Input("anndata-upload-test", "contents"),
        Input("anndata-upload-app", "contents"),
    ]
)
def update_status(x_uploaded, train_uploaded, test_uploaded, app_uploaded):
    x_style = {"display": "inline"} if x_uploaded else {"display": "none"}
    train_style = {"display": "inline"} if train_uploaded else {"display": "none"}
    test_style = {"display": "inline"} if test_uploaded else {"display": "none"}
    app_style = {"display": "inline"} if app_uploaded else {"display": "none"}
    import_disabled = not (x_uploaded and train_uploaded and test_uploaded)
    return x_style, train_style, test_style, app_style, import_disabled

@callback(
    Output('upload-file', 'contents'),
    Output('upload-file', 'filename'),
    Output("import-anndata-modal", "opened", allow_duplicate=True),
    Input("anndata-import-btn", "n_clicks"),
    State("anndata-upload-x", "contents"),
    State("anndata-upload-train", "contents"),
    State("anndata-upload-test", "contents"),
    State("anndata-upload-app", "contents"),
    prevent_initial_call=True
)
def ImportFromAnndata(n_clicks, X_binary, Train_binary, Test_binary, App_binary):
    dcx = DeconomixFile.from_AnnData(X_binary, Train_binary, Test_binary, App_binary, description="Imported from AnnData")

    return dcx.to_contents_string(), 'ImportedFromAnnData', False

@callback(
    Output('upload-modal-train', 'opened'),
    Output('upload-modal-train-diagram', 'children'),
    Input('upload-button-training-distribution', 'n_clicks'),
    State('upload-modal-train', 'opened'),
    prevent_initial_call=True
)
def showTrainDistribution(trainDistributionClicked, opened):
    global localDCXCache

    distribution_plot = dmc.Text("Here should be a diagram")

    if localDCXCache.DeconomixFile.Train is not None:
        distribution_plot = get_distribution_plot(localDCXCache.DeconomixFile.Train)

    return not opened, distribution_plot


@callback(
    Output('upload-modal-test', 'opened'),
    Output('upload-modal-test-diagram', 'children'),
    Input('upload-button-testing-distribution', 'n_clicks'),
    State('upload-modal-test', 'opened'),
    prevent_initial_call=True
)
def showTestDistribution(testDistributionClicked, opened):
    global localDCXCache

    distribution_plot = dmc.Text("Here should be a diagram")

    if localDCXCache.DeconomixFile.Test is not None:
        distribution_plot = get_distribution_plot(localDCXCache.DeconomixFile.Test)

    return not opened, distribution_plot


@callback(
    Output('upload-modal-reference', 'opened'),
    Input('upload-button-reference', 'n_clicks'),
    State('upload-modal-reference', 'opened'),
    prevent_initial_call=True
)
def showReferenceMatrix(referenceClicked, opened):
    return not opened


### Callbacks DTD ###
@callback(
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
    # This shall load all parameters, check them and then trigger the loading screen, which itself should trigger DTD
    global localDCXCache

    # Check if button was pressed
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


@callback(
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
    global localDCXCache

    if "dtd-exec-overlay" == ctx.triggered_id and exec_overlay_visible is True:
        print(nGenes)
        print("Simulating training data")
        _, localDCXCache.DTD_Y_train, localDCXCache.DTD_C_train = simulate_data(localDCXCache.DeconomixFile.Train,
                                                                                localDCXCache.DTD_config.train_n_mixtures,
                                                                                localDCXCache.DTD_config.train_n_cells,
                                                                                n_genes=nGenes)

        print("Simulating testing data")
        _, localDCXCache.DTD_Y_test, localDCXCache.DTD_C_test = simulate_data(localDCXCache.DeconomixFile.Test,
                                                                              localDCXCache.DTD_config.test_n_mixtures,
                                                                              localDCXCache.DTD_config.test_n_cells)

        # Ensure genes of test and train data are the same
        localDCXCache.DTD_Y_test = localDCXCache.DTD_Y_test.loc[localDCXCache.DTD_Y_train.index,:]

        localDCXCache.DTDmodel = DTD(localDCXCache.DeconomixFile.X_mat.loc[localDCXCache.DTD_Y_train.index,:],
                                     localDCXCache.DTD_Y_train,
                                     localDCXCache.DTD_C_train)

        localDCXCache.DTDmodel.run(localDCXCache.DTD_config.nIter)

        # Estimate Compositions
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

        # Load the tabs with the result visualization
        dtd_tab_loss = get_tab_dtd_loss(localDCXCache)
        dtd_tab_corr = get_tab_dtd_correlation(localDCXCache)
        dtd_tab_mix = get_tab_dtd_mixture(localDCXCache, runOnApplChecked)
        dtd_tab_marker = get_tab_dtd_markermap(localDCXCache)

        return False, False, dtd_tab_loss, dtd_tab_corr, dtd_tab_mix, dtd_tab_marker, False
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update


@callback(
    Output("tab-dtd-scatter-train", "data"),
    Output("tab-dtd-scatter-test", "data"),
    Output("tab-dtd-train-corr", "children"),
    Output("tab-dtd-test-corr", "children"),
    Input("tab-dtd-correlation-combo", "value"),
    Input("tab-dtd-correlation-combo", "data")
)
def DTDselectCorrPlot(celltype, combobox_items):
    global localDCXCache

    data_train = get_correlation_dict(localDCXCache.DTD_C_train.loc[combobox_items[int(celltype[2]) - 1]['label']],
                                      localDCXCache.DTD_C_train_est.loc[combobox_items[int(celltype[2]) - 1]['label']],
                                      name='Training')
    data_test = get_correlation_dict(localDCXCache.DTD_C_test.loc[combobox_items[int(celltype[2]) - 1]['label']],
                                     localDCXCache.DTD_C_test_est.loc[combobox_items[int(celltype[2]) - 1]['label']],
                                     name='Testing')

    corr_train, _ = spearmanr(localDCXCache.DTD_C_train.loc[combobox_items[int(celltype[2]) - 1]['label']],
                              localDCXCache.DTD_C_train_est.loc[combobox_items[int(celltype[2]) - 1]['label']])

    corr_val, _ = spearmanr(localDCXCache.DTD_C_test.loc[combobox_items[int(celltype[2]) - 1]['label']],
                            localDCXCache.DTD_C_test_est.loc[combobox_items[int(celltype[2]) - 1]['label']])

    return data_train, data_test, \
        dmc.Text(f"Correlation: {np.round(corr_train, 4)}", ta="center"), \
        dmc.Text(f"Correlation: {np.round(corr_val, 4)}", ta="center")


@callback(
    Output("dtd-mix-mixture-combo", "data"),
    Input("dtd-mix-dataset-combo", "value")
)
def DTDselectMixtureDataSet(dataset):
    global localDCXCache

    if dataset == "train":
        selected_data = localDCXCache.DTD_C_train_est
    elif dataset == "test":
        selected_data = localDCXCache.DTD_C_test_est
    elif dataset == "appl":
        selected_data = localDCXCache.DTD_C_appl_est

    mixture_data = fill_combo_mixtures(selected_data)

    return mixture_data


@callback(
    Output("dtd-mix-pie-plot", "figure"),
    Input("dtd-mix-dataset-combo", "value"),
    Input("dtd-mix-mixture-combo", "value")
)
def UpdateDTDPiePlot(selectedDataset, selectedMixture):
    global localDCXCache

    if selectedDataset == "train":
        selected_data = localDCXCache.DTD_C_train_est
    elif selectedDataset == "test":
        selected_data = localDCXCache.DTD_C_test_est
    elif selectedDataset == "appl":
        selected_data = localDCXCache.DTD_C_appl_est

    mixture = selected_data.iloc[:, int(selectedMixture[1:]) - 1]

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

    return pie_plot


@callback(
    Output("dtd-mix-dataset-download", "data"),
    Input("dtd-mix-dataset-button-download", "n_clicks"),
    State("dtd-mix-dataset-combo", "value")
)
def DTDDownloadEstimate(n_clicks, selectedDataset):
    global localDCXCache
    # Check if button was pressed
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


@callback(
    Output("dtd-markermap-plot", "children"),
    Input("dtd-markermap-n-genes", "value")
)
def UpdateMarkermapPlot(nGenes):
    global localDCXCache

    markerplot = get_markermap_plot(localDCXCache, nGenes)

    return markerplot


@callback(
    Output("dtd-gamma-download", "data"),
    Input("dtd-gamma-button-download", "n_clicks")
)
def DownloadGamma(n_clicks):
    global localDCXCache

    if "dtd-gamma-button-download" == ctx.triggered_id:
        gamma = localDCXCache.DTDmodel.gamma.T.iloc[0]
        return dcc.send_data_frame(gamma.to_csv, f"gamma.csv")
    else:
        return None


@callback(
    Input('dtd-skeleton', 'visible'),
    State("main-content", "children")
)
def storeCurrentDTDTab(skeletonVisible, currDTDTab):
    global localDCXCache
    localDCXCache.DTDTab = currDTDTab



### Callbacks ADTD ###
@callback(
    Output('adtd-par-lambda1', 'disabled'),
    Input('adtd-par-check-Cstatic', 'checked'),
    prevent_initial_call=True
)
def updateLambda1Input(CstaticChecked):

    if CstaticChecked: return True
    else: return False

@callback(
    Output('adtd-par-lambda2', 'disabled'),
    Input('adtd-par-check-Deltastatic', 'checked'),
    prevent_initial_call=True
)
def updateLambda2Input(DeltastaticChecked):

    if DeltastaticChecked: return True
    else: return False


@callback(
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
    global localDCXCache

    localDCXCache.ADTD_config.lambda2min = lambda_min
    localDCXCache.ADTD_config.lambda2max = lambda_max
    localDCXCache.ADTD_config.nPoints = npoints
    localDCXCache.ADTD_config.Dataset = dataset

    if dataset == 'train':
        Y_hps = localDCXCache.DTD_Y_train
    elif dataset == 'appl':
        Y_hps = localDCXCache.DeconomixFile.Application.loc[localDCXCache.DTDmodel.gamma.index,:] # Ensure that application has the same genes as DTD model
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

    return dcc.Graph(figure=fig), 'hps', False, True, True, optimalLambda2 #gr_disabled, mix_disabled


@callback(
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
    global localDCXCache

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
        Y_adtd = localDCXCache.DeconomixFile.Application.loc[localDCXCache.DTDmodel.gamma.index,:] # Ensure that application has the same genes as DTD model    
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

    return tab_mixture, tab_gr, 'mixtures', hps_disabled, Deltastatic, False


@callback(
    Output("adtd-mix-pie-plot", "figure"),
    Input("adtd-mix-mixture-combo", "value"),
    prevent_initial_call=True
)
def UpdateADTDPiePlot(selectedMixture):
    global localDCXCache

    selected_data = localDCXCache.ADTDmodel.C_est.copy()
    selected_data.loc['hidden'] = localDCXCache.ADTDmodel.c_est.iloc[0]

    mixture = selected_data.iloc[:, int(selectedMixture[1:]) - 1]

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

    return pie_plot

@callback(
    Output("adtd-mix-dataset-download", "data"),
    Input("adtd-mix-dataset-button-download", "n_clicks"),
    State("adtd-dataset-combo", "value")
)
def ADTDDownloadEstimate(n_clicks, selectedDataset):
    global localDCXCache
    # Check if button was pressed
    if "adtd-mix-dataset-button-download" == ctx.triggered_id:
        selected_data = localDCXCache.ADTDmodel.C_est.copy()
        selected_data.loc['hidden'] = localDCXCache.ADTDmodel.c_est.iloc[0]

        return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_ADTD_est.csv")
    else:
        return None


@callback(
    Output("adtd-gr-selected-genes", "error"),
    Output("adtd-gr-plot", "figure"),
    Input("adtd-gr-selected-genes", "value"),
    State("adtd-gr-plot", "figure")
)
def UpdateGeneRegulationPlot(genes, curr_fig):
    global localDCXCache

    error = ""
    fig = curr_fig
    if len(genes) < 1:
        error = "Select at least one gene!"
    else:
        fig = get_gr_plot(localDCXCache, sorted(genes, key=str.lower))

    return error, fig

@callback(
    Output("adtd-gr-download", "data"),
    Input("adtd-gr-button-download", "n_clicks"),
    State("adtd-dataset-combo", "value")
)
def ADTDDownloadGR(n_clicks, selectedDataset):
    global localDCXCache
    # Check if button was pressed
    if "adtd-gr-button-download" == ctx.triggered_id:
        selected_data = localDCXCache.ADTDmodel.Delta_est

        return dcc.send_data_frame(selected_data.to_csv, f"{selectedDataset}_Delta.csv")
    else:
        return None

@callback(
    Output("adtd-skeleton", "visible"),
    Input("adtd-dataset-combo", "value"),
    State("adtd-skeleton", "visible"),
)
def resetResults(selectedDataset, skeletonVisible):
    global localDCXCache

    if localDCXCache.ADTD_config.Dataset == selectedDataset:
        return skeletonVisible
    else:
        localDCXCache.clearADTD()
        return True


@callback(
    Input('adtd-skeleton', 'visible'),
    State("main-content", "children")
)
def storeCurrentADTDTab(skeletonVisible, currADTDTab):
    global localDCXCache
    localDCXCache.ADTDTab = currADTDTab


@callback(
    Output("dcxconvert-modal", "opened"),
    [Input("dcxconvert-trigger", "n_clicks"),
     Input("dcxconvert-cancel-btn", "n_clicks")],
    [State("dcxconvert-modal", "opened")],
    prevent_initial_call=True
)
def toggle_dcxconvert_modal(open_click, cancel_click, modal_open):
    ctx_id = ctx.triggered_id
    if ctx_id == "dcxconvert-trigger":
        return True
    elif ctx_id == "dcxconvert-cancel-btn":
        return False
    return modal_open

@callback(
    [Output("dcxconvert-indicator-x", "style"),
     Output("dcxconvert-indicator-train", "style"),
     Output("dcxconvert-indicator-test", "style"),
     Output("dcxconvert-indicator-app", "style"),
     Output("dcxconvert-download-btn", "disabled")],
    [Input("dcxconvert-upload-x", "contents"),
     Input("dcxconvert-upload-train", "contents"),
     Input("dcxconvert-upload-test", "contents"),
     Input("dcxconvert-upload-app", "contents"),
     Input("dcxconvert-author", "value"),
     Input("dcxconvert-filename", "value")]
)
def update_dcxconvert_indicators(x, train, test, app, author, filename):
    x_style = {"display": "inline"} if x else {"display": "none"}
    train_style = {"display": "inline"} if train else {"display": "none"}
    test_style = {"display": "inline"} if test else {"display": "none"}
    app_style = {"display": "inline"} if app else {"display": "none"}
    enable = bool(x and train and test and author and filename)
    return x_style, train_style, test_style, app_style, not enable


@callback(
    Output("dcxconvert-download", "data"),
    Input("dcxconvert-download-btn", "n_clicks"),
    State("dcxconvert-upload-x", "contents"),
    State("dcxconvert-upload-train", "contents"),
    State("dcxconvert-upload-test", "contents"),
    State("dcxconvert-upload-app", "contents"),
    State("dcxconvert-author", "value"),
    State("dcxconvert-desc", "value"),
    State("dcxconvert-filename", "value"),
    State("dcxconvert-appdesc", "value"),
    State("dcxconvert-traindesc", "value"),
    State("dcxconvert-testdesc", "value"),
    prevent_initial_call=True
)
def dcxconvert_download(n_clicks, x, train, test, app, author, desc, filename, appdesc, traindesc, testdesc):
    if not (n_clicks and x and train and test and author and filename):
        return None
    def decode_csv(content):
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.BytesIO(decoded), header=0, index_col=0)
    X_mat = decode_csv(x)
    Train = decode_csv(train)
    Test = decode_csv(test)
    Application = decode_csv(app) if app else None
    deconomix_file = DeconomixFile(
        X_mat=X_mat,
        Train=Train,
        Test=Test,
        Application=Application,
        description=desc or "",
        author=author,
        filename=filename
    )
    deconomix_file.TrainDesc = traindesc or "Dataset used for Training"
    deconomix_file.TestDesc = testdesc or "Dataset used for Testing"
    deconomix_file.ApplicationDesc = appdesc or "Bulk data for Application"
    # Serialisieren als Bytes
    b64str = deconomix_file.to_contents_string()
    # Extrahiere nur den base64-Teil
    b64 = b64str.split(",", 1)[-1]
    file_bytes = base64.b64decode(b64)
    return dcc.send_bytes(lambda buf: buf.write(file_bytes), f"{filename}.dcx")


server = app.server

if __name__ == "__main__":
    app.title = "DeconomiX"

    app.run(debug=True)
    #app.run_server(debug=False)
