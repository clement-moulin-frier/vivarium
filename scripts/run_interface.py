import argparse

from vivarium.interface.panel_app import WindowManager

parser = argparse.ArgumentParser(description='Run the Vivarium interface.')
parser.add_argument('notebook_mode', action='store_false', help='Run in notebook mode.')
args = parser.parse_args()

# Serve the app to launch the interface
wm = WindowManager(notebook_mode=args.notebook_mode)
wm.app.servable(title="Vivarium")
