# DeconomiX GUI

DeconomiX GUI is a modular, extensible, and user-friendly web application for deconvolution analysis of omics data. It is designed for both researchers and non-expert users, supporting plug-in-based workflows, robust dependency management, and future extensibility.
<br>
Visit our website: https://deconomix.bioinf.med.uni-goettingen.de

## Features

- **Plug-in Architecture:** Easily add or remove analysis modules (pages) as plug-ins. Each plug-in is auto-discovered and can specify dependencies, order, and navigation details via a `.config` file.
- **User-Friendly Interface:** Modern, intuitive GUI built with Dash and Mantine Components, suitable for non-technical users.
- **Extensible:** Ready for new analysis types (e.g., preprocessing, survival analysis) via plug-ins.
- **Multi-User Ready:** Designed for safe, session-based multi-user/server scenarios (see notes below).
- **Internationalization:** All navigation and help texts are in English, ready for global use.

## Getting Started

### Prerequisites
- Python 3.13.2 (this version was used for developement)
- Recommended: Create a virtual environment


## Plug-in System
- Each plug-in is a Python module in the `pages/` directory with a corresponding `.config` file.
- Example `.config` fields:
  ```
  position = 1
  id = nav-dtd_page
  title = DTD Analysis
  description = Perform classical deconvolution (DTD) and evaluate the results.
  dependencies = Uploading
  ```
- To add a new plug-in, create a new Python file and a `.config` file in `pages/`.
- Required functions: `get_layout(app)`, `register_callbacks(app)`, `nav_disabled(session_id)`.

### Accessing Data from the Session Cache

- Use the function `get_session_cache(session_id)` (from `utils/session_cache_manager`) to access a per-session cache object (`DCXCache`).
- All plugins and callbacks must receive the `session_id` (provided via `dcc.Store` and Dash callback `State('session-id', 'data')`) and use it to access or store data.
- Example usage in a plug-in:
  ```python
  from utils.session_cache_manager import get_session_cache
  # session_id must be provided by the Dash callback State
  cache = get_session_cache(session_id)
  # Access uploaded file
  if hasattr(cache, 'DeconomixFile'):
      df = cache.DeconomixFile.X_mat  # or other attributes
  # Store results
  cache.DTDmodel = my_dtd_result
  ```

## License
See [LICENSE](LICENSE) for details.

