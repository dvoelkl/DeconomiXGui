# DeconomiX GUI

DeconomiX GUI is a modular, extensible, and user-friendly web application for deconvolution analysis of omics data. It is designed for both researchers and non-expert users, supporting plug-in-based workflows, robust dependency management, and future extensibility.

## Features

- **Plug-in Architecture:** Easily add or remove analysis modules (pages) as plug-ins. Each plug-in is auto-discovered and can specify dependencies, order, and navigation details via a `.config` file.
- **User-Friendly Interface:** Modern, intuitive GUI built with Dash and Mantine Components, suitable for non-technical users.
- **Extensible:** Ready for new analysis types (e.g., preprocessing, survival analysis) via plug-ins.
- **Multi-User Ready:** Designed for safe, session-based multi-user/server scenarios (see notes below).
- **Internationalization:** All navigation and help texts are in English, ready for global use.

## Getting Started

### Prerequisites
- Python 3.9+
- Recommended: Create a virtual environment

### Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd DeconomiXGui
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the App
```sh
python DeconomixApp.py
```
The app will start a local web server. Open the displayed URL in your browser.

## Usage
- **Uploading:** Upload your data and select analysis parameters.
- **DTD Analysis:** Perform classical deconvolution (DTD) and evaluate results. (Requires data upload.)
- **ADTD Analysis:** Advanced deconvolution with additional options. (Requires DTD and data upload.)
- Navigation is dynamic: Only available plug-ins and those with fulfilled dependencies are shown.
- If a plug-in cannot be loaded due to missing dependencies, a notification will appear.

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
- Required functions: `get_layout(app)` and `register_callbacks(app)`.

### Accessing Data from the Session Cache

- The legacy global object `localDCXCache` (from `utils/global_cache`) has been **deprecated** and replaced by a robust, session-aware cache system.
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
- **Best practice:** Only store session-specific, non-sensitive data in the session cache. For multi-user/server scenarios, the cache is now properly isolated per user/session.

## Multi-Session Architecture
- The app now uses a `SessionCacheManager` (see `utils/session_cache_manager.py`) to manage isolated cache objects for each session/user/project.
- All data (uploads, models, results) are kept strictly per session. Plugins and callbacks must always use the session-aware cache.
- The old `utils/global_cache.py` is deprecated and should not be used.
- For details and extension ideas (e.g., auto-cleanup, encryption), see `MultiCacheExtension.md`.

## Multi-User & Security Notes
- The app is now robust for multi-user/server scenarios: session and cache handling is isolated and secure by design.
- Uploaded data and analysis results are kept per session. No data is shared between sessions unless explicitly implemented.
- Review your deployment and infrastructure for additional security or compliance needs.

## Development & Contribution
- Contributions are welcome! Please document new plug-ins and keep `.config` files up to date.
- For bug reports or feature requests, open an issue.

## License
See [LICENSE](LICENSE) for details.

---

**Contact:** For questions or support, please contact the project maintainer.
