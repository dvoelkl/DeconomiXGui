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
import plotly.graph_objects as go

import matplotlib
import numpy as np

import base64
import io
import pandas as pd
import scanpy as sc

import configparser

# Imports for Deconomix
from utils.DeconomixFile import DeconomixFile
from utils.DTD_config import DTDConfig
from utils.ADTD_config import ADTDConfig
from utils.DeconomixCache import DCXCache
from scipy.stats import spearmanr

# Import from Deconomix
from deconomix.utils import simulate_data, calculate_estimated_composition
from deconomix.methods import DTD
from utils.session_cache_manager import get_session_cache, session_manager

import importlib
import os
import inspect
import uuid

#########################  #########################

######################### GLOBALS #########################
# Session-based cache logic
# session_cache = get_session_cache()  # REMOVED: No global session cache in multi-session architecture

#########################  #########################


######################### GUI #########################

_dash_renderer._set_react_version("18.2.0")

app = Dash(external_stylesheets=dmc.styles.ALL, suppress_callback_exceptions=True)

logo_filename = "DeconomiX_Logo.png"

######################### PLUGIN-SYSTEM #########################

# Automatic plug-in discovery with .config evaluation and dependency resolution
PLUGINS = []
PLUGIN_ERRORS = []
REGISTERED = set()
PAGES_PATH = os.path.join(os.path.dirname(__file__), "pages")
PLUGIN_CONFIGS = []

# First, collect all configs
for fname in os.listdir(PAGES_PATH):
    if fname.endswith(".py") and not fname.startswith("_"):
        modname = fname[:-3]
        config_path = os.path.join(PAGES_PATH, f"{modname}.config")
        config = {"modname": modname}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        config[key.strip()] = value.strip().strip('"')
        PLUGIN_CONFIGS.append(config)

# Register plug-ins in dependency order
while PLUGIN_CONFIGS:
    progress = False
    for config in PLUGIN_CONFIGS[:]:
        modname = config["modname"]
        dependencies = [d.strip() for d in config.get("dependencies", "").split(",") if d.strip()]
        if all(dep in REGISTERED or dep == modname for dep in dependencies):
            module = importlib.import_module(f"pages.{modname}")
            if hasattr(module, "get_layout") and hasattr(module, "register_callbacks"):
                position = int(config.get("position", 999))
                nav_id = config.get("id", f"nav-{modname.lower()}")
                title = config.get("title", modname.replace("_page", "").capitalize())
                description = config.get("description", f"{title} plug-in")
                PLUGINS.append({
                    "position": position,
                    "label": title,
                    "description": description,
                    "id": nav_id,
                    "module": module,
                })
                REGISTERED.add(modname)
            PLUGIN_CONFIGS.remove(config)
            progress = True
    if not progress:
        # Could not resolve some plug-ins due to missing dependencies
        for config in PLUGIN_CONFIGS:
            modname = config["modname"]
            dependencies = [d.strip() for d in config.get("dependencies", "").split(",") if d.strip()]
            missing = [dep for dep in dependencies if dep not in REGISTERED and dep != modname]
            if missing:
                PLUGIN_ERRORS.append(f"Plug-in '{modname}' could not be registered. Missing dependencies: {', '.join(missing)}.")
        break

# Sort by position
PLUGINS.sort(key=lambda x: x["position"])

# Dynamic navigation

def get_nav_links(session_id=None, last_selected_tab=None):
    """Erzeuge NavLinks, wobei DTD- und ADTD-Tab nur aktiviert werden, wenn Voraussetzungen erfüllt sind."""
    dtd_disabled = True
    adtd_disabled = True
    #if session_id:
    #    try:
    #        cache = get_session_cache(session_id)
    #        #dtd_disabled = getattr(cache, "DeconomixFile", None) is None
    #        # ADTD-Tab nur aktivieren, wenn Datei geladen UND DTDmodel existiert
    #        adtd_disabled = getattr(cache, "DeconomixFile", None) is None or getattr(cache, "DTDmodel", None) is None
    #    except Exception:
    #        dtd_disabled = True
    #        adtd_disabled = True
    links = []
    for plugin in PLUGINS:
        is_disabled = False
        if hasattr(module, "nav_disabled") and callable(plugin["module"].nav_disabled):
            is_disabled = plugin["module"].nav_disabled(session_id)

        links.append(dmc.NavLink(label=plugin["label"], description=plugin["description"], id=plugin["id"], disabled=is_disabled))

        #if plugin["id"] == "nav-dtd_page":
        #    links.append(dmc.NavLink(label=plugin["label"], description=plugin["description"], id=plugin["id"], disabled=dtd_disabled))
        #elif plugin["id"] == "nav-adtd_page":
        #    links.append(dmc.NavLink(label=plugin["label"], description=plugin["description"], id=plugin["id"], disabled=adtd_disabled))
        #else:
        #    links.append(dmc.NavLink(label=plugin["label"], description=plugin["description"], id=plugin["id"]))
    return links

# Notification area for dependency errors
notification_area = []
if PLUGIN_ERRORS:
    notification_area = [
        dmc.Alert(
            title="Plug-in registration error",
            color="red",
            children=[html.Ul([html.Li(msg) for msg in PLUGIN_ERRORS])],
            style={"marginBottom": 20}
        )
    ]

layout = dmc.AppShell(
    notification_area + [
        dcc.Store(id="session-id", data=str(uuid.uuid4()), storage_type="session"),
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
            children=get_nav_links(),
            p="md",
        ),
        dmc.AppShellMain(id="main-content",
                         children=PLUGINS[0]["module"].get_layout() if PLUGINS else dmc.Text("No plug-ins available.")),
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
    Output("navbar", "children", allow_duplicate=True),
    [Input(plugin["id"], "n_clicks") for plugin in PLUGINS],
    State("session-id", "data"),
    prevent_initial_call=True,
)
def display_plugin(*args):
    ctx_id = ctx.triggered_id
    session_id = args[-1]  # session_id from dcc.Store
    nav_links = get_nav_links(session_id)
    layout = None
    for plugin in PLUGINS:
        if plugin["id"] == ctx_id:
            layout = plugin["module"].display_plugin(session_id)
            nav_links = get_nav_links(session_id, ctx_id)
            return layout, nav_links    
            return plugin["module"].get_layout(session_id), nav_links

    if layout is None:  # Fallback to first plugin if no match found
        layout = PLUGINS[0]["module"].get_layout(session_id)
    return layout, nav_links

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

# Callback: Generate session-id if not set
@callback(
    Output('session-id', 'data'),
    Input('session-id', 'data'),
    prevent_initial_call=True
)
def ensure_session_id(session_id):
    if not session_id:
        new_id = str(uuid.uuid4())
        return new_id
    return session_id

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
