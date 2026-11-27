"""Model names and research levels; safe to import without third-party packages."""
BASELINES = ('persistence', 'input_mean')
NEW_BASELINES = ('seasonal_climatology', 'seasonal_ar1', 'ridge_var')
LEVEL0 = (*BASELINES, 'sarima', *NEW_BASELINES)
LEVEL1 = ('mlp', 'tcn', 'gru')
LEVEL2 = ('nbeats', 'nhits', 'patchtst', 'dlinear', 'tide')
NEURAL_MODELS = (*LEVEL1, *LEVEL2)
ALL_MODELS = (*LEVEL0, *NEURAL_MODELS)


def model_level(name):
    """Return the experimental level associated with the model name."""
    for level, names in enumerate((LEVEL0, LEVEL1, LEVEL2)):
        if name in names:
            return level
    return None
