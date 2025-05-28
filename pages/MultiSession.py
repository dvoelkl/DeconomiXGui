# MultiSession management page for DeconomiX GUI
# All code and comments in English

import dash_mantine_components as dmc
from dash import html, dcc, Output, Input, State, callback, no_update, ALL
from utils.session_cache_manager import session_manager

# --- Layout ---
def get_layout(session_id=None):
    sessions = session_manager.list_sessions()
    archived = session_manager.list_archived_sessions()

    # Sort: Aktive Session immer ganz oben
    sessions_sorted = []
    if session_id:
        active = [s for s in sessions if s['session_id'] == session_id]
        others = [s for s in sessions if s['session_id'] != session_id]
        sessions_sorted = active + others
    else:
        sessions_sorted = sessions

    def session_row(s, archived=False):
        is_active = (s['session_id'] == session_id)
        # Zeige den Namen aus den Metadaten, nicht die session_id
        session_name = s.get('name', f"Session {s['session_id'][:8]}")
        return dmc.Paper([
            dmc.Group([
                dmc.Text(session_name, fw=700 if is_active else 400, c='teal' if is_active else 'default', style={"minWidth": 140}),
                dmc.Text(s['session_id'], size='xs', c='dimmed', style={"minWidth": 120}),
                dmc.Text(s.get('created', ''), size='xs', c='dimmed', style={"minWidth": 120}),
                dmc.Text(s.get('status', 'active'), size='xs', c='green' if not archived else 'orange', style={"minWidth": 70}),
                dmc.Group([
                    dmc.Button('Switch', id={'type': 'switch-session', 'index': s['session_id']}, size='xs', variant='filled' if is_active else 'outline', color='teal' if is_active else 'gray', disabled=is_active),
                    dmc.Button('Rename', id={'type': 'rename-session', 'index': s['session_id']}, size='xs', color='blue'),
                    dmc.Button('Archive' if not archived else 'Restore', id={'type': 'archive-session', 'index': s['session_id']}, size='xs', color='orange'),
                    dmc.Button('Delete', id={'type': 'delete-session', 'index': s['session_id']}, size='xs', color='red', variant='outline', disabled=is_active),
                ], gap='xs'),
            ], gap='md', align='center'),
        ], shadow='xs', radius='md', p=8, mb=6, style={'backgroundColor': '#e6fcf5' if is_active else 'white', 'border': '1px solid #dee2e6'})

    return dmc.Stack([
        dmc.Title('Session Manager', order=2),
        dmc.Text(f'Active session: {session_id}', c='teal'),
        dmc.Button('Create new session', id='create-session-btn', color='green', mt=10),
        dmc.Divider(label='Active Sessions', my=10),
        dmc.Stack([session_row(s) for s in sessions_sorted]),
        dmc.Divider(label='Archived Sessions', my=10),
        dmc.Stack([session_row(s, archived=True) for s in archived]),
        dcc.ConfirmDialog(id='confirm-delete-session', message='Do you really want to permanently delete this session?'),
        dcc.Store(id='session-manager-action'),
        dcc.Store(id='session-manager-target'),
        dmc.Modal(
            id='rename-session-modal',
            title='Rename Session',
            opened=False,
            children=[
                dmc.TextInput(id='rename-session-input', label='New session name', value=''),
                dmc.Group([
                    dmc.Button('Cancel', id='rename-session-cancel', color='gray'),
                    dmc.Button('Rename', id='rename-session-confirm', color='blue'),
                ], mt=10)
            ]
        ),
        dmc.Notification(id='session-manager-notify', title='Session Manager', message='', color='teal', autoClose=2000, style={'display': 'none'}, action=None),
    ])

# --- Callbacks ---
def register_callbacks(app):
    from dash import Output, Input, State, ctx, no_update, ALL
    import dash_mantine_components as dmc
    from dash import dcc
    import logging

    # Callback: Create new session
    @app.callback(
        Output('main-content', 'children', allow_duplicate=True),
        Output('navbar', 'children', allow_duplicate=True),
        Output('session-id', 'data', allow_duplicate=True),
        Output('session-manager-notify', 'message', allow_duplicate=True),
        Output('session-manager-notify', 'style', allow_duplicate=True),
        Input('create-session-btn', 'n_clicks'),
        State('session-id', 'data'),
        State('main-content', 'children'),
        prevent_initial_call=True
    )
    def create_session(n_clicks, session_id, main_content):
        from dash import callback_context
        ctx = callback_context
        if not n_clicks or not ctx.triggered:
            return no_update, no_update, no_update, '', {'display': 'none'}
        new_id = session_manager.create_session()
        #layout = get_layout(new_id)
        # Debug: Log relevant information
        print(f"[DEBUG] Create session clicked. Current session_id: {session_id}, New session_id: {new_id}")

        active_plugin = 'nav-uploading'
        try:
            if isinstance(main_content, dict) and 'props' in main_content and 'id' in main_content['props']:
                from DeconomixApp import PLUGINS
                for plugin in PLUGINS:
                    if plugin['id'] in str(main_content['props']['id']):
                        active_plugin = plugin['id']
                        break
        except Exception:
            pass

        from DeconomixApp import PLUGINS, display_plugin, get_nav_links
        plugin_args = [None] * len(PLUGINS)
        for i, plugin in enumerate(PLUGINS):
            if plugin['id'] == active_plugin:
                plugin_args[i] = 1  # Simuliere Klick
        plugin_args.append(new_id)
        layout_result = display_plugin(*plugin_args)
        # Falls display_plugin ein Tupel (layout, nav_links) liefert, beide extrahieren
        if isinstance(layout_result, (tuple, list)) and len(layout_result) > 1:
            layout, nav_links = layout_result[0], layout_result[1]
        elif isinstance(layout_result, (tuple, list)) and len(layout_result) > 0:
            layout, nav_links = layout_result[0], get_nav_links(new_id)
        else:
            layout, nav_links = layout_result, get_nav_links(new_id)

        return layout, nav_links, new_id, f"Created new session {new_id}", {'display': 'block'}

    # Callback: Switch session
    @app.callback(
        Output('main-content', 'children', allow_duplicate=True),
        Output('navbar', 'children', allow_duplicate=True),
        Output('session-id', 'data', allow_duplicate=True),
        Output('session-manager-notify', 'message', allow_duplicate=True),
        Output('session-manager-notify', 'style', allow_duplicate=True),
        Input({'type': 'switch-session', 'index': ALL}, 'n_clicks'),
        State('session-id', 'data'),
        State('main-content', 'children'),
        prevent_initial_call=True
    )
    def switch_session(n_clicks_list, session_id, main_content):
        from dash import callback_context
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return no_update, no_update, no_update, '', {'display': 'none'}
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        target_id = eval(triggered)  # {'type': 'switch-session', 'index': ...}
        print(f"[DEBUG] Switch-Session Event: triggered by {triggered}")
        print(f"[DEBUG] Current session_id: {session_id}")
        print(f"[DEBUG] Target session_id: {target_id['index']}")
        new_id = target_id['index']
        if new_id == session_id:
            return no_update, no_update, no_update, '', {'display': 'none'}
        # Ermittle aktives Plugin anhand des aktuellen main_content (Fallback: DTD)
        active_plugin = 'nav-dtd_page'
        try:
            # Suche nach Plugin-ID im aktuellen Content (z.B. DTD, ADTD, Uploading)
            if isinstance(main_content, dict) and 'props' in main_content and 'id' in main_content['props']:
                from DeconomixApp import PLUGINS
                for plugin in PLUGINS:
                    if plugin['id'] in str(main_content['props']['id']):
                        active_plugin = plugin['id']
                        break
        except Exception:
            pass
        from DeconomixApp import PLUGINS, display_plugin, get_nav_links
        plugin_args = [None] * len(PLUGINS)
        for i, plugin in enumerate(PLUGINS):
            if plugin['id'] == active_plugin:
                plugin_args[i] = 1  # Simuliere Klick
        plugin_args.append(new_id)
        layout_result = display_plugin(*plugin_args)
        # Falls display_plugin ein Tupel (layout, nav_links) liefert, beide extrahieren
        if isinstance(layout_result, (tuple, list)) and len(layout_result) > 1:
            layout, nav_links = layout_result[0], layout_result[1]
        elif isinstance(layout_result, (tuple, list)) and len(layout_result) > 0:
            layout, nav_links = layout_result[0], get_nav_links(new_id)
        else:
            layout, nav_links = layout_result, get_nav_links(new_id)
        return layout, nav_links, new_id, f'Switched to session {new_id}', {'display': 'block'}

    # Callback: Open rename modal
    @app.callback(
        Output('rename-session-modal', 'opened', allow_duplicate=True),
        Output('rename-session-input', 'value', allow_duplicate=True),
        Output('session-manager-target', 'data', allow_duplicate=True),
        Input({'type': 'rename-session', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def open_rename_modal(n_clicks_list):
        from dash import callback_context
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return False, '', no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        target_id = eval(triggered)
        return True, '', target_id['index']

    # Callback: Confirm rename
    @app.callback(
        Output('main-content', 'children', allow_duplicate=True),
        Output('rename-session-modal', 'opened', allow_duplicate=True),
        Output('session-manager-notify', 'message', allow_duplicate=True),
        Output('session-manager-notify', 'style', allow_duplicate=True),
        Output('session-id', 'data', allow_duplicate=True),
        Input('rename-session-confirm', 'n_clicks'),
        State('rename-session-input', 'value'),
        State('session-manager-target', 'data'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def confirm_rename(n_clicks, new_name, target_id, session_id):
        if not n_clicks or not new_name or not target_id:
            return no_update, False, '', {'display': 'none'}, no_update
        session_manager.rename_session(target_id, new_name)
        layout = get_layout(session_id)
        return layout, False, f'Session renamed to {new_name}', {'display': 'block'}, session_id

    # Callback: Cancel rename
    @app.callback(
        Output('rename-session-modal', 'opened', allow_duplicate=True),
        Input('rename-session-cancel', 'n_clicks'),
        prevent_initial_call=True
    )
    def cancel_rename(n_clicks):
        return False

    # Callback: Archive/Restore session
    @app.callback(
        Output('main-content', 'children', allow_duplicate=True),
        Output('session-manager-notify', 'message', allow_duplicate=True),
        Output('session-manager-notify', 'style', allow_duplicate=True),
        Output('session-id', 'data', allow_duplicate=True),
        Input({'type': 'archive-session', 'index': ALL}, 'n_clicks'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def archive_restore_session(n_clicks_list, session_id):
        from dash import callback_context
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return no_update, '', {'display': 'none'}, no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        target_id = eval(triggered)
        sid = target_id['index']
        # Check if session is archived or active
        archived_ids = [s['session_id'] for s in session_manager.list_archived_sessions()]
        if sid in archived_ids:
            session_manager.restore_session(sid)
            msg = f'Session {sid} restored.'
        else:
            if sid == session_id:
                return no_update, 'Cannot archive active session.', {'display': 'block'}, no_update
            session_manager.archive_session(sid)
            msg = f'Session {sid} archived.'
        layout = get_layout(session_id)
        return layout, msg, {'display': 'block'}, session_id

    # Callback: Open delete confirm dialog
    @app.callback(
        Output('confirm-delete-session', 'displayed', allow_duplicate=True),
        Output('session-manager-target', 'data', allow_duplicate=True),
        Input({'type': 'delete-session', 'index': ALL}, 'n_clicks'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def open_delete_confirm(n_clicks_list, session_id):
        from dash import callback_context
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return False, no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        target_id = eval(triggered)
        sid = target_id['index']
        if sid == session_id:
            return False, no_update
        return True, sid

    # Callback: Confirm delete
    @app.callback(
        Output('main-content', 'children', allow_duplicate=True),
        Output('confirm-delete-session', 'displayed', allow_duplicate=True),
        Output('session-manager-notify', 'message', allow_duplicate=True),
        Output('session-manager-notify', 'style', allow_duplicate=True),
        Output('session-id', 'data', allow_duplicate=True),
        Input('confirm-delete-session', 'submit_n_clicks'),
        State('session-manager-target', 'data'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def confirm_delete(n_clicks, target_id, session_id):
        if not n_clicks or not target_id:
            return no_update, False, '', {'display': 'none'}, no_update
        session_manager.delete_session(target_id)
        # Nach Löschen ggf. auf eine andere Session wechseln
        sessions = session_manager.list_sessions()
        new_id = sessions[0]['session_id'] if sessions else None
        layout = get_layout(new_id)
        return layout, False, f'Session {target_id} deleted.', {'display': 'block'}, new_id
