####################################################
#
# Uploading page for DeconomiX GUI application
#
####################################################

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, _dash_renderer, dcc, html
import pandas as pd
from utils.DeconomixFile import DeconomixFile

def get_upload_layout():
    upload_layout = dmc.Stack([
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
                    'margin': '10px'
                },
                multiple=False
            ),
            dmc.Button("Import from AnnData", id="anndata-trigger-import"),
            dmc.Modal(
                title="File format not supported",
                id="upload-file-not-supported-alert",
                children=dmc.Text("The uploaded file could not be parsed as DeconomiX File!"),
            ),
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
                                    'textAlign': 'center', 'margin': '10px'
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
                                    'textAlign': 'center', 'margin': '10px'
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
                                    'textAlign': 'center', 'margin': '10px'
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
                                    'textAlign': 'center', 'margin': '10px'
                                },
                            ),
                        ]),
                        dmc.Group([
                            dmc.Button("Import", id="anndata-import-btn", disabled=True, color="blue"),
                            dmc.Button("Cancel", id="anndata-cancel-btn", color="red", variant="outline"),
                        ])
                    ])
                ],
            )

        ]),
        dmc.Skeleton(
                    visible=False,
                    children=html.Div(id="upload-file-properties"),
                    mb=10
                )
        ]
    )

    return upload_layout

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
                "Reference Matrix used for DeconomiX",
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
    #file = DeconomixFile()

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
                                dmc.Text(size="sm", mb=5, children="DeconomiX file version:"),
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

    n_celltypes = scData.columns.value_counts()
    data = []

    for celltype in n_celltypes.index:
        celltype_to_append = {"celltype": celltype, "count": n_celltypes[celltype]}
        data.append(celltype_to_append)


    distribution_plot = dmc.BarChart(
            h=300,
            dataKey="celltype",
            data=data,
            type="stacked",
            series=[
                {"name": "count", "color": "blue.6"}
            ],
        )

    return distribution_plot