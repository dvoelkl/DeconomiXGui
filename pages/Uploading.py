####################################################
#
# Uploading page for Deconomix GUI application
#
####################################################

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, _dash_renderer, dcc, html
import pandas as pd
from utils.DeconomixFile import DeconomixFile
from utils.session_cache_manager import get_session_cache, session_manager
import base64, io
import scanpy as sc
import numpy as np
import dash


def get_indicator_style(uploaded):
    return {"display": "inline"} if uploaded else {"display": "none"}

def all_required_fields_filled(*fields):
    return all(fields)

def decode_input(content):
    if content is None:
        return None
    header = content[:30]
    if header.startswith("data:application/octet-stream"):  # AnnData (h5ad)
        _, b64data = content.split(",", 1)
        decoded = base64.b64decode(b64data)
        with io.BytesIO(decoded) as f:
            adata = sc.read_h5ad(f)
            return adata.to_df() if hasattr(adata, 'to_df') else pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    else:  # CSV
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.BytesIO(decoded), header=0, index_col=0)

def get_layout(session_id=None):
    cache = get_session_cache(session_id) if session_id else None
    # File-Properties vorbereiten, falls Datei geladen
    file_properties = None
    if cache and getattr(cache, 'DeconomixFile', None) is not None:
        file = cache.DeconomixFile
        filename = getattr(file, 'filename', '')
        modification_date = ''
        file_properties = get_file_properties_layout(filename, modification_date, file)

    layout = dmc.Stack([
        html.Div([
            dcc.Upload(
                id='upload-file',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px auto'
                },
                multiple=False
            ),
            dmc.Group([
                dmc.Button("Import from AnnData", id="anndata-trigger-import"),
                dmc.Button("DCX Converter", id="dcxconvert-trigger", color="teal", ml=10),
            ]),
            dmc.Modal(
                title="Import from AnnData",
                id="import-anndata-modal",
                opened=False,
                children=[
                    dmc.Stack([
                        dmc.Stack(children=[
                            # Reference profile
                            dmc.Text("Reference profile"),
                            dmc.Badge("Uploaded", id="anndata-indicator-x", color="green", variant="outline",
                                      style={"display": "none"}),
                            dcc.Upload(
                                id="anndata-upload-x",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                                multiple=False
                            ),]),
                        dmc.Stack(children=[
                            # Train data
                            dmc.Text("Train data"),
                            dmc.Badge("Uploaded", id="anndata-indicator-train", color="green", variant="outline",
                                      style={"display": "none"}),
                            dcc.Upload(
                                id="anndata-upload-train",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                            ),]),
                        dmc.Stack(children=[
                            # Test data
                            dmc.Text("Test data"),
                            dmc.Badge("Uploaded", id="anndata-indicator-test", color="green", variant="outline",
                                      style={"display": "none"}),
                            dcc.Upload(
                                id="anndata-upload-test",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                            ),]),
                        dmc.Stack(children=[
                            # Application data (Optional)
                            dmc.Text("Application data (Optional)"),
                            dmc.Badge("Uploaded", id="anndata-indicator-app", color="green", variant="outline",
                                      style={"display": "none"}),
                            dcc.Upload(
                                id="anndata-upload-app",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                            ),
                        ]),
                        dmc.Group([
                            dmc.Button("Import", id="anndata-import-btn", disabled=True, color="blue"),
                            dmc.Button("Cancel", id="anndata-cancel-btn", color="red", variant="outline"),
                        ])
                    ])
                ],
            ),
            dmc.Modal(
                title="Create DCX File",
                id="dcxconvert-modal",
                opened=False,
                size="lg",
                styles={"modal": {"width": 700}},
                children=[
                    dmc.Stack([
                        dmc.Text(
                            "Convert data to the Deconomix .dcx format. Supported input types: CSV (with header and index) or AnnData (.h5ad) files. Required fields: Reference Profile X, Single Cell Train Data, Single Cell Test Data. Optional: Application Data. All fields accept either CSV or AnnData.",
                            size="sm",
                            mb=10
                        ),
                        dmc.Group([
                            dmc.Text("Author*", w=120),
                            dmc.TextInput(id="dcxconvert-author", placeholder="Author", style={"flex":1}),
                        ], gap="md"),
                        dmc.Group([
                            dmc.Text("File Description", w=120),
                            dmc.TextInput(id="dcxconvert-desc", placeholder="Description", style={"flex":1}),
                        ], gap="md"),
                        dmc.Group([
                            dmc.Text("Filename*", w=120),
                            dmc.TextInput(id="dcxconvert-filename", placeholder="Filename without .dcx", style={"flex":1}),
                        ], gap="md"),
                        dmc.Divider(label="Required", my=10),
                        dmc.Stack([
                            dmc.Text("Reference Profile X*"),
                            dmc.Badge("Uploaded", id="dcxconvert-indicator-x", color="green", variant="outline", style={"display": "none"}),
                            dcc.Upload(
                                id="dcxconvert-upload-x",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                                multiple=False
                            ),
                        ]),
                        dmc.Stack([
                            dmc.Text("Single Cell Train Data*"),
                            dmc.Badge("Uploaded", id="dcxconvert-indicator-train", color="green", variant="outline", style={"display": "none"}),
                            dcc.Upload(
                                id="dcxconvert-upload-train",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                                multiple=False
                            ),
                        ]),
                        dmc.Stack([
                            dmc.Text("Single Cell Test Data*"),
                            dmc.Badge("Uploaded", id="dcxconvert-indicator-test", color="green", variant="outline", style={"display": "none"}),
                            dcc.Upload(
                                id="dcxconvert-upload-test",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'auto'
                                },
                                multiple=False
                            ),
                        ]),
                        dmc.Divider(label="Optional Fields", my=10),
                        dmc.Stack([
                            dmc.Text("Bulk Application Data (optional)"),
                            dmc.Badge("Uploaded", id="dcxconvert-indicator-app", color="green", variant="outline", style={"display": "none"}),
                            dcc.Upload(
                                id="dcxconvert-upload-app",
                                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                    'textAlign': 'center', 'margin': 'a'
                                },
                                multiple=False
                            ),
                        ]),
                        dmc.Group([
                            dmc.Text("Application Description", w=180),
                            dmc.TextInput(id="dcxconvert-appdesc", placeholder="Description of application data", style={"flex":1}),
                        ], gap="md"),
                        dmc.Group([
                            dmc.Text("Train Description", w=180),
                            dmc.TextInput(id="dcxconvert-traindesc", placeholder="Description of train data", style={"flex":1}),
                        ], gap="md"),
                        dmc.Group([
                            dmc.Text("Test Description", w=180),
                            dmc.TextInput(id="dcxconvert-testdesc", placeholder="Description of test data", style={"flex":1}),
                        ], gap="md"),
                        dmc.Group([
                            dmc.Button("Download", id="dcxconvert-download-btn", color="green", disabled=True),
                            dmc.Button("Cancel", id="dcxconvert-cancel-btn", color="red", variant="outline"),
                        ], style={"justifyContent": "flex-end", "gap": 10}, mt=10),
                        dcc.Download(id="dcxconvert-download"),
                        dmc.Text(id="dcxconvert-error-text", style={"color": "red", "minHeight": 24, "marginTop": 4, "marginBottom": 4}, size="sm"),
                    ])
                ]
            ),
        ]),
        dmc.Skeleton(
                    visible=False,
                    children=html.Div(id="upload-file-properties", children=file_properties),
                    mb=10
                ),
        dmc.Modal(
            title="File format not supported",
            id="upload-file-not-supported-alert",
            children=dmc.Text("The uploaded file could not be parsed as Deconomix File!"),
        )
        ]
    )
    return layout

def register_callbacks(app):
    from dash import Output, Input, State, no_update, ctx
    import dash_mantine_components as dmc
    global get_file_properties_layout, get_distribution_plot
    from utils.DeconomixFile import DeconomixFile

    def upload_deconomix_file_callback(content, filename, last_modified, dtd_was_disabled, adtd_was_disabled, session_id):
        cache = get_session_cache(session_id)
        showAlert = False
        if cache is None:
            disabled_dtd = True
            disabled_adtd = True
            file_properties_layout = "No file loaded!"
            return file_properties_layout, disabled_dtd, disabled_adtd, showAlert
        if filename is None and getattr(cache, 'DeconomixFile', None) is None:
            disabled_dtd = True
            disabled_adtd = True
            file_properties_layout = "No file loaded!"
            cache.DeconomixFile = None
        else:
            disabled_dtd = False
            disabled_adtd = True
            if filename is not None:
                try:
                    _, content = content.split(',')
                    content = io.BytesIO(base64.b64decode(content))
                    cache.DeconomixFile = DeconomixFile.unbinarize(content)
                    cache.DeconomixFile.filename = filename
                    file_properties_layout = get_file_properties_layout(filename, last_modified, cache.DeconomixFile)
                    cache.clearAll()
                    # Persistiere Session nach erfolgreichem Upload
                    session_manager.save_session(session_id)
                except Exception as e:
                    showAlert = True
                    disabled_dtd = True
                    disabled_adtd = True
                    file_properties_layout = "No file loaded!"
                    cache.DeconomixFile = None
            else:
                file_properties_layout = get_file_properties_layout(cache.DeconomixFile.filename, "", cache.DeconomixFile)
            if getattr(cache, 'DTDmodel', None) is not None:
                disabled_adtd = False
        return file_properties_layout, disabled_dtd, disabled_adtd, showAlert

    @app.callback(
        Output('upload-file-properties', 'children'),
        Output('upload-file-not-supported-alert', 'opened', allow_duplicate=True),
        Output('navbar', 'children', allow_duplicate=True),
        Input('upload-file', 'contents'),
        Input('upload-file', 'filename'),
        Input('upload-file', 'last_modified'),
        State('session-id', 'data'),
        prevent_initial_call='initial_duplicate')
    def uploadDeconomixFile(content, filename, last_modified, session_id):
        from DeconomixApp import get_nav_links
        file_properties_layout, _, _, showAlert = upload_deconomix_file_callback(content, filename, last_modified, None, None, session_id)
        nav_links = get_nav_links(session_id)
        return file_properties_layout, showAlert, nav_links

    @app.callback(
        Output("dcxconvert-modal", "opened", allow_duplicate=True),
        [Input("dcxconvert-trigger", "n_clicks"),
         Input("dcxconvert-cancel-btn", "n_clicks")],
        [State("dcxconvert-modal", "opened"),
         State('session-id', 'data')],
        prevent_initial_call=True
    )
    def toggle_dcxconvert_modal(open_click, cancel_click, modal_open, session_id):
        from dash import ctx
        print(f"DCX Convert Modal triggered by: {ctx.triggered_id}")
        ctx_id = ctx.triggered_id
        if ctx_id == "dcxconvert-trigger":
            return True
        elif ctx_id == "dcxconvert-cancel-btn":
            return False
        return modal_open

    @app.callback(
        Output("import-anndata-modal", "opened", allow_duplicate=True),
        Input("anndata-trigger-import", "n_clicks"),
        Input("anndata-cancel-btn", "n_clicks"),
        State("import-anndata-modal", "opened"),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def toggle_anndata_modal(open_click, cancel_click, modal_open, session_id):
        print(f"AnnData Import Modal triggered by: {ctx.triggered_id}")
        ctx_id = ctx.triggered_id
        if ctx_id == "anndata-trigger-import":
            return True
        elif ctx_id == "anndata-cancel-btn":
            return False
        return modal_open

    @app.callback(
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
         Input("dcxconvert-filename", "value"),
         State('session-id', 'data')]
    )
    def update_dcxconvert_indicators(x, train, test, app_, author, filename, session_id):
        x_style = {"display": "inline"} if x else {"display": "none"}
        train_style = {"display": "inline"} if train else {"display": "none"}
        test_style = {"display": "inline"} if test else {"display": "none"}
        app_style = {"display": "inline"} if app_ else {"display": "none"}
        disabled = not (x and train and test and author and filename)
        return x_style, train_style, test_style, app_style, disabled

    @app.callback(
        Output("dcxconvert-download", "data"),
        Output("dcxconvert-modal", "opened", allow_duplicate=True),
        Output("dcxconvert-error-text", "children"),
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
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def dcxconvert_download(n_clicks, x, train, test, app_, author, desc, filename, appdesc, traindesc, testdesc, session_id):
        print(f"DCX Convert Download triggered by: {ctx.triggered_id}")
        if not (n_clicks and x and train and test and author and filename):
            return dash.no_update, dash.no_update, "Please fill all required fields."
        try:
            # Use decode_input for all uploads (AnnData or CSV)
            deconomix_file = DeconomixFile(
                X_mat=decode_input(x),
                Train=decode_input(train),
                Test=decode_input(test),
                Application=decode_input(app_),
                description=desc,
                author=author,
                filename=filename
            )
            deconomix_file.TrainDesc = traindesc or "Dataset used for Training"
            deconomix_file.TestDesc = testdesc or "Dataset used for Testing"
            deconomix_file.ApplicationDesc = appdesc or "Bulk data for Application"
            cache = get_session_cache(session_id)
            cache.DeconomixFile = deconomix_file
            b64str = deconomix_file.to_contents_string()
            b64 = b64str.split(",", 1)[-1]
            file_bytes = base64.b64decode(b64)
            return dcc.send_bytes(lambda buf: buf.write(file_bytes), f"{filename}.dcx"), False, ""
        except Exception as e:
            return dash.no_update, dash.no_update, f"Error: {str(e)}"

    @app.callback(
        [Output("anndata-indicator-x", "style"),
         Output("anndata-indicator-train", "style"),
         Output("anndata-indicator-test", "style"),
         Output("anndata-indicator-app", "style"),
         Output("anndata-import-btn", "disabled")],
        [Input("anndata-upload-x", "contents"),
         Input("anndata-upload-train", "contents"),
         Input("anndata-upload-test", "contents"),
         Input("anndata-upload-app", "contents"),
         State('session-id', 'data')],
    )
    def update_status(x_uploaded, train_uploaded, test_uploaded, app_uploaded, session_id):
        x_style = {"display": "inline"} if x_uploaded else {"display": "none"}
        train_style = {"display": "inline"} if train_uploaded else {"display": "none"}
        test_style = {"display": "inline"} if test_uploaded else {"display": "none"}
        app_style = {"display": "inline"} if app_uploaded else {"display": "none"}
        import_disabled = not (x_uploaded and train_uploaded and test_uploaded)
        return x_style, train_style, test_style, app_style, import_disabled

    @app.callback(
        Output('upload-file', 'contents'),
        Output('upload-file', 'filename'),
        Output("import-anndata-modal", "opened", allow_duplicate=True),
        Input("anndata-import-btn", "n_clicks"),
        State("anndata-upload-x", "contents"),
        State("anndata-upload-train", "contents"),
        State("anndata-upload-test", "contents"),
        State("anndata-upload-app", "contents"),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def ImportFromAnndata(n_clicks, X_binary, Train_binary, Test_binary, App_binary, session_id):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        cache = get_session_cache(session_id)
        dcx = DeconomixFile.from_AnnData(X_binary, Train_binary, Test_binary, App_binary, description="Imported from AnnData")
        cache.DeconomixFile = dcx
        return dcx.to_contents_string(), 'ImportedFromAnnData', False

    @app.callback(
        Output('upload-modal-train', 'opened'),
        Output('upload-modal-train-diagram', 'children'),
        Input('upload-button-training-distribution', 'n_clicks'),
        State('upload-modal-train', 'opened'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def showTrainDistribution(trainDistributionClicked, opened, session_id):
        cache = get_session_cache(session_id)
        distribution_plot = dmc.Text("Here should be a diagram")
        if cache.DeconomixFile.Train is not None:
            distribution_plot = get_distribution_plot(cache.DeconomixFile.Train)
        return not opened, distribution_plot

    @app.callback(
        Output('upload-modal-test', 'opened'),
        Output('upload-modal-test-diagram', 'children'),
        Input('upload-button-testing-distribution', 'n_clicks'),
        State('upload-modal-test', 'opened'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def showTestDistribution(testDistributionClicked, opened, session_id):
        cache = get_session_cache(session_id)
        distribution_plot = dmc.Text("Here should be a diagram")
        if cache.DeconomixFile.Test is not None:
            distribution_plot = get_distribution_plot(cache.DeconomixFile.Test)
        return not opened, distribution_plot

    @app.callback(
        Output('upload-modal-reference', 'opened'),
        Input('upload-button-reference', 'n_clicks'),
        State('upload-modal-reference', 'opened'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def showReferenceMatrix(referenceClicked, opened, session_id):
        return not opened

def nav_disabled(session_id):
    return False # Should always be enabled

def display_plugin(session_id):
    return get_layout(session_id)

def get_card_reference():
    # Card Reference
    card_ref = dmc.Card(
        id="upload-card-reference",
        children=[
            dmc.CardSection(
                dmc.Image(
                    src="assets/background3.jpeg",
                    h=160,
                    alt="ImageForReference",
                )
            ),
            dmc.Text("Reference Matrix", fw=500),
            dmc.Text(
                "Reference Matrix used for Deconomix",
                size="sm",
                c="dimmed",
            ),
            #dmc.Button(
            #    "View Table",
            #    color="blue",
            #    fullWidth=True,
            #    mt="md",
            #    radius="md",
            #    id="upload-button-reference"
            #),
            dmc.Modal(
                title="Reference Matrix",
                id="upload-modal-reference",
                children=[
                    dmc.Text("Here should be a table (Not implemented yet)"),
                ]
            )
        ],
        withBorder=True,
        shadow="sm",
        radius="md",
        w=350,
    )

    return card_ref

def get_card_train(train_desc):

    card_train = dmc.Card(
        id="upload-card-training",
        children=[
            dmc.CardSection(
                dmc.Image(
                    src="assets/background1.jpeg",
                    h=160,
                    alt="ImageForTraining",
                )
            ),
            dmc.Text("Training Data", fw=500),
            dmc.Text(
                f"{train_desc}",
                size="sm",
                c="dimmed",
            ),
            dmc.Button(
                "Display Distribution",
                color="blue",
                fullWidth=True,
                mt="md",
                radius="md",
                id="upload-button-training-distribution"
            ),
            dmc.Modal(
                title="Training Distribution",
                id="upload-modal-train",
                children=[
                    html.Div(id="upload-modal-train-diagram")
                ]
            )
        ],
        withBorder=True,
        shadow="sm",
        radius="md",
        w=350,
    )

    return card_train

def get_card_test(test_desc):
    card_test = dmc.Card(id="upload-card-testing",
                        children=[
                            dmc.CardSection(
                                dmc.Image(
                                    src="assets/background2.jpeg",
                                    h=160,
                                    alt="ImageForTest",
                                )
                            ),
                            dmc.Text("Testing Data", fw=500),
                            dmc.Text(
                                f"{test_desc}",
                                size="sm",
                                c="dimmed",
                            ),
                            dmc.Button(
                                "Display Distribution",
                                color="blue",
                                fullWidth=True,
                                mt="md",
                                radius="md",
                                id="upload-button-testing-distribution"
                            ),
                             dmc.Modal(
                                 title="Testing Distribution",
                                 id="upload-modal-test",
                                 children=[
                                     html.Div(id="upload-modal-test-diagram")
                                 ]
                             )
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                        w=350,
                    )

    return card_test

def get_card_app(app_desc):
    card_app = dmc.Card(
                        id="upload-card-application",
                        children=[
                            dmc.CardSection(
                                dmc.Image(
                                    src="assets/background3.jpeg",
                                    h=160,
                                    alt="ImageForApplication",
                                )
                            ),
                            dmc.Text("Application Data", fw=500),
                            dmc.Text(
                                f"{app_desc}",
                                size="sm",
                                c="dimmed",
                            ),
                            #dmc.Button(
                            #    "Display Details",
                            #    color="blue",
                            #    fullWidth=True,
                            #    mt="md",
                            #    radius="md",
                            #    id="upload-button-application-information"
                            #),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                        w=350,
                    )

    return card_app

def get_card_gamma():
    card_gamma = dmc.Card(
                        id="upload-card-gamma",
                        children=[
                            dmc.CardSection(
                                dmc.Image(
                                    src="assets/background1.jpeg",
                                    h=160,
                                    alt="ImageForGamma",
                                )
                            ),
                            dmc.Text("Gamma", fw=500),
                            dmc.Text(
                                "Calculated Gamma Vector by DTD",
                                size="sm",
                                c="dimmed",
                            ),
                            #dmc.Button(
                            #    "Display Details",
                            #    color="blue",
                            #    fullWidth=True,
                            #    mt="md",
                            #    radius="md",
                            #    id="upload-button-gamma-information"
                            #),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                        w=350,
                    )

    return card_gamma

def get_file_properties_layout(filename, modification_date, file):

    CardDisplay = []
    if file.X_mat is not None:
        CardDisplay.append(get_card_reference())

    if file.Train is not None:
        CardDisplay.append(get_card_train(file.TrainDesc))

    if file.Test is not None:
        CardDisplay.append(get_card_test(file.TestDesc))

    if file.Application is not None:
        CardDisplay.append(get_card_app(file.ApplicationDesc))

    if file.Gamma is not None:
        CardDisplay.append(get_card_gamma(file.Gamma))

    file_properties_layout = dmc.Accordion(
        variant="contained",
        chevronPosition="left",
        children=[
            dmc.AccordionItem(
                [
                    dmc.AccordionControl("General file properties"),
                    dmc.AccordionPanel(
                        dmc.SimpleGrid(
                            cols=2,
                            spacing="25px",
                            verticalSpacing="md",
                            children=[
                                dmc.Text(size="sm", mb=5, children="File Name:"),
                                dmc.Text(size="sm", mb=5, children=f"{filename}"),
                                dmc.Text(size="sm", mb=5, children="Deconomix file version:"),
                                dmc.Text(size="sm", mb=5, children=f"{file.__VERSION__}"),
                                dmc.Text(size="sm", mb=5, children="Created by:"),
                                dmc.Text(size="sm", mb=5, children=f"{file.Author}")
                            ]
                        )
                    )
                ],
                value="general_properties"
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl("Contained Data"),
                    dmc.AccordionPanel(
                        dmc.Group(
                            CardDisplay
                        )
                    )
                ],
                value="data"
            )
        ]
    )

    return file_properties_layout

def get_distribution_plot(scData):

    import plotly.express as px
    import dash_mantine_components as dmc
    import dash
    import pandas as pd
    
    if isinstance(scData, pd.DataFrame):
        if hasattr(scData, 'columns') and hasattr(scData.columns, 'value_counts'):
            celltype_counts = scData.columns.value_counts()
        else:
            celltype_counts = scData.iloc[:,0].value_counts()
    elif isinstance(scData, pd.Series):
        celltype_counts = scData.value_counts()
    else:
        return dmc.Text("Invalid data for distribution plot.")

    fig = px.bar(
        x=celltype_counts.index,
        y=celltype_counts.values,
        labels={'x': 'Cell Type', 'y': 'Count'},
        title='Cell Type Distribution'
    )
    return dcc.Graph(figure=fig)