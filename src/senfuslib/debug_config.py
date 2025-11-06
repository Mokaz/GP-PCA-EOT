from pathlib import Path

DEBUG = False and __debug__  # set to False for speedup

sim_output_dir = Path(__file__).parents[1] / 'data' / 'cache'
sim_output_dir.mkdir(exist_ok=True, parents=True)

# TODO Martin : Figure out if this file is needed