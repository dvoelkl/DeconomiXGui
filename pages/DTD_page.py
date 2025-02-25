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
from utils.DeconomixCache import DCXCache
import numpy as np

def get_dtd_layout(applCheckEnabled):

    dtd_layout = dmc.Stack(
        [
            # Loading
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
                visible=True,
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
                                                  html.Div("Loss function display", id="dtd-res-loss")
                                              ]),

                                dmc.TabsPanel(value="correlation",
                                              children=[
                                                  html.Div("Correlation plots", id="dtd-res-corr")
                                              ]),

                                dmc.TabsPanel(value="markermap",
                                              children=[
                                                  html.Div("Markermap", id="dtd-res-markermap")
                                              ]),

                                dmc.TabsPanel(value="mixtures",
                                              children=[
                                                  html.Div("Estimated Mixture plots", id="dtd-res-mixtures")
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