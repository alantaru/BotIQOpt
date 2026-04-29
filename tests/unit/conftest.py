import matplotlib

# Use a non-interactive backend for matplotlib to prevent it from
# trying to create a GUI window, which can cause tests to hang.
matplotlib.use('Agg')
