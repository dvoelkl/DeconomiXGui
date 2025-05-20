####################################################
#
# Main entry point for DeconomiX GUI application
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
import scanpy as sc


# Imports for Deconomix
from utils.DeconomixFile import DeconomixFile
from utils.DTD_config import DTDConfig
from utils.ADTD_config import ADTDConfig
from utils.DeconomixCache import DCXCache
from scipy.stats import spearmanr

# Import from Deconomix
from deconomix.utils import simulate_data, calculate_estimated_composition
from deconomix.methods import DTD
from utils.global_cache import localDCXCache

#########################  #########################

######################### GLOBALS #########################
# localDCXCache is now imported from utils.global_cache

#########################  #########################


######################### GUI #########################

_dash_renderer._set_react_version("18.2.0")

app = Dash(external_stylesheets=dmc.styles.ALL, suppress_callback_exceptions=True)

logo_filename = "DeconomiX_Logo.png"

######################### PLUGIN-SYSTEM #########################

from pages import DTD_page, ADTD_page, Uploading

PLUGINS = [
    {
        "label": "Load",
        "description": "Load a DeconomiX file",
        "id": "nav-load",
        "module": Uploading,
    },
    {
        "label": "DTD",
        "description": "Run DTD",
        "id": "nav-dtd",
        "module": DTD_page,
    },
    {
        "label": "ADTD",
        "description": "Run ADTD",
        "id": "nav-adtd",
        "module": ADTD_page,
    },
    # Additional plug-ins can be added here
]

# Dynamic navigation
nav_links = [
    dmc.NavLink(label=plugin["label"], description=plugin["description"], id=plugin["id"]) for plugin in PLUGINS
]

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
            children=nav_links,
            p="md",
        ),
        dmc.AppShellMain(id="main-content",
                         children=Uploading.get_layout()),
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

# Plug-in callback registration
for plugin in PLUGINS:
    if hasattr(plugin["module"], "register_callbacks"):
        plugin["module"].register_callbacks(app)

# Dynamic routing for page display
@callback(
    Output("main-content", "children", allow_duplicate=True),
    [Input(plugin["id"], "n_clicks") for plugin in PLUGINS],
    prevent_initial_call=True,
)
def display_plugin(*args):
    ctx_id = ctx.triggered_id
    for plugin in PLUGINS:
        if plugin["id"] == ctx_id:
            return plugin["module"].get_layout()
    return Uploading.get_layout()

### Callbacks Navigation ###
@callback(
    Output("appshell", "navbar", allow_duplicate=True),
    Input("mobile-burger", "opened"),
    Input("desktop-burger", "opened"),
    State("appshell", "navbar"),
    prevent_initial_call=True,
)
def toggle_navbar(mobile_opened, desktop_opened, navbar):
    navbar["collapsed"] = {
        "mobile": not mobile_opened,
        "desktop": not desktop_opened,
    }
    return navbar


######################### HELPER FUNCTIONS #########################

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

def get_indicator_style(uploaded):
    """Helper function for green indicator style."""
    return {"display": "inline"} if uploaded else {"display": "none"}

def all_required_fields_filled(*fields):
    """Checks if all required fields are filled."""
    return all(fields)

def handle_decode_input(content):
    """Error handling for decode_input."""
    try:
        return decode_input(content)
    except Exception as e:
        return f"Error while parsing: {str(e)}"

def check_gene_index_consistency(*dfs):
    """Checks if all DataFrames have the same row indices."""
    sets = [set(df.index) for df in dfs if df is not None]
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            if sets[i] != sets[j]:
                return False, i, j
    return True, None, None

######################### CALLBACKS #########################

# All upload and modal callbacks have been moved to the Uploading plug-in.

server = app.server

if __name__ == "__main__":
    app.title = "DeconomiX"

    app.run(debug=True)
    #app.run_server(debug=False)
