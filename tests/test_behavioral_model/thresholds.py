"""How large an effect a perturbation must produce, and how small it may stay.
"""

# Mean probability shift a full min-to-max group perturbation must produce.
# Measured: congestion +0.022 ('all') and +0.033 ('noweather'); weather +0.035.
MIN_CONGESTION_EFFECT = 0.005
MIN_WEATHER_EFFECT = 0.005

# Worst recorded conditions versus mildest, as a ratio rather than an absolute
# floor: the baseline depends on the variant and on the fold.
# Measured: 1.36 ('all') and 1.18 ('noweather').
MIN_SEPARATION_RATIO = 1.05

# The flight number is an identifier the pipeline happens to feed as a numeric
# feature. Airlines do assign number ranges by route type, so a small effect is
# legitimate; a large one means the model is keying on the identity itself.
MAX_FLIGHT_NUMBER_EFFECT = 0.10

# A constant predictor would satisfy every bound above. Measured spread: ~0.035.
MIN_PREDICTION_SPREAD = 0.01

# Calibration is fitted before registration, so on held-out flights the mean
# predicted probability should land near the rate actually observed.
MAX_CALIBRATION_GAP = 0.15
