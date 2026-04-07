"""
DetectorConfig – loads the ML model once at startup via AppConfig.ready().
The loaded model is stored as `DetectorConfig.model` for use in views.
"""

import os
import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    # Shared model instance (populated in ready())
    model  = None
    device = "cpu"

    def ready(self):
        """Load model weights once the Django app registry is initialised."""
        from django.conf import settings

        weights_path = getattr(settings, "MODEL_WEIGHTS_PATH", None)
        model_type   = getattr(settings, "MODEL_TYPE", "3dcnn")
        device       = getattr(settings, "MODEL_DEVICE", "cpu")

        if not weights_path or not os.path.isfile(weights_path):
            logger.warning(
                "MODEL_WEIGHTS_PATH is not set or file does not exist at '%s'. "
                "Inference will return an error until weights are placed there.",
                weights_path,
            )
            return

        try:
            from detector.ml.inference import load_model
            DetectorConfig.model  = load_model(weights_path, model_type, device)
            DetectorConfig.device = device
            logger.info("Shoplifting detection model loaded (%s on %s).", model_type, device)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
