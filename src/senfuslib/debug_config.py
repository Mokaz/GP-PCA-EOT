from src.global_project_paths import PROJECT_ROOT

DEBUG = False and __debug__  # set to False for speedup
SHOW_PROGRESSBAR = True

sim_output_dir = PROJECT_ROOT / 'data' / 'cache'
sim_output_dir.mkdir(exist_ok=True, parents=True)

# TODO Martin : Figure out if this file is needed