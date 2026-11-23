"""Live (prospective) next-day forecasting on the strict WSTS input contract.

Everything here is offline: a scheduled script builds today's strict input frame from live
sources, runs the trained fold checkpoints and writes static GeoTIFFs. The backend only serves
those files (it never imports this package or torch).

Design rule: live frames are assembled as 23-band *raw* WSTS frames and then passed through
``fireguard_ml.data.wsts_dataset.process_raw``. No strict channel is ever computed here, so the
transforms a model sees live are, by construction, the ones it was trained with.
"""
