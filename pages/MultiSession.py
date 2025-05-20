# MultiSession management page for DeconomiX GUI
# All code and comments in English

import dash_mantine_components as dmc
from dash import html, dcc, Output, Input, State, callback, no_update
from utils.session_cache_manager import session_manager

# --- Layout ---
def get_layout(session_id=None):
    sessions = session_manager.list_sessions()
    archived = session_manager.list_archived_sessions()
    
    def session_row(s, archived=False):
        is_active = (s['session_id'] == session_id)
        return dmc.Paper([
            dmc.Group([
                dmc.Text(s.get('name', s['session_id']), fw=700 if is_active else 400, c='teal' if is_active else 'default', style={"minWidth": 140}),
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
        dmc.Stack([session_row(s) for s in sessions]),
        dmc.Divider(label='Archived Sessions', my=10),
        dmc.Stack([session_row(s, archived=True) for s in archived]),
        dcc.ConfirmDialog(id='confirm-delete-session', message='Do you really want to permanently delete this session?'),
        dcc.Store(id='session-manager-action'),
        dcc.Store(id='session-manager-target'),
        dmc.Notification(id='session-manager-notify', title='Session Manager', message='', color='teal', autoClose=2000, style={'display': 'none'}, action=None),
    ])

# --- Callbacks ---
def register_callbacks(app):
    # Placeholders for actual callback logic (to be implemented in next steps)
    pass
