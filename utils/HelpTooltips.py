####################################################
#
# File for handling help tooltips in the GUI
#
####################################################


class HelpTooltipsManager:
 
    def get_tooltip_content(self, id):
        tooltip = self.tooltip_dict.get(id, "No tooltip available for this element.")
        
        return tooltip

    def register_tooltip(self, id, content):
        """Registers a new tooltip for a given element ID."""
        if id not in self.tooltip_dict:
            self.tooltip_dict[id] = content
        

    def register_tooltip_config_file(self, config_file):
        if config_file is not None and config_file != "":
            try:
                with open(config_file, 'r') as file:
                    for line in file:
                        if '=' in line:
                            id, content = line.split('=', 1)
                            self.register_tooltip(id.strip(), content.strip())
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
            except Exception as e:
                raise RuntimeError(f"Error reading tooltip config file: {e}")

    def __init__(self, delayTime=1000):
        self.tooltip_dict = {}
        self.delayTime = delayTime # in milliseconds

def get_HelpTooltipsManager():
    """Returns the singleton instance of HelpTooltipsManager."""
    return help_tooltips

# Singleton instance for global access
help_tooltips = HelpTooltipsManager()
 