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

### Accessing Data from `localDCXCache`
- The global object `localDCXCache` (imported from `utils.global_cache`) is used to store and share data (such as uploaded files, models, and results) between plug-ins during a session.
- Example usage in a plug-in:
  ```python
  from utils.global_cache import localDCXCache
  # Access uploaded file
  if hasattr(localDCXCache, 'DeconomixFile'):
      df = localDCXCache.DeconomixFile.X_mat  # or other attributes
  # Store results
  localDCXCache.DTDmodel = my_dtd_result
  ```
- **Best practice:** Only store session-specific, non-sensitive data in `localDCXCache`. For multi-user/server scenarios, ensure that the cache is properly isolated per user/session.

## Multi-User & Security Notes
- The app is designed for multi-user scenarios, but you should ensure that session and cache handling is secure for your deployment.
- Uploaded data and analysis results are kept per session, but review the code and adapt for your infrastructure if needed.

## Development & Contribution
- Contributions are welcome! Please document new plug-ins and keep `.config` files up to date.
- For bug reports or feature requests, open an issue.

## License
See [LICENSE](LICENSE) for details.

---

**Contact:** For questions or support, please contact the project maintainer.
