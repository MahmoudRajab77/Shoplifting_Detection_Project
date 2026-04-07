"""
Views for the Shoplifting Detector Django app.

Endpoints:
  GET  /           → Upload form (IndexView)
  POST /predict/   → Process uploaded video, show result page (PredictView)
  POST /api/predict/ → JSON API endpoint (PredictAPIView)
"""

import os
import uuid
import json
import logging

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .forms import VideoUploadForm
from .ml.video_utils import load_video_frames
from .ml.inference import predict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
def _get_model():
    """Return the singleton model loaded by DetectorConfig.ready()."""
    from detector.apps import DetectorConfig
    return DetectorConfig.model, DetectorConfig.device


def _run_inference(video_file) -> dict:
    """Save upload → extract frames → predict. Returns result dict."""
    # Save to MEDIA_ROOT/uploads/<uuid>.<ext>
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    ext      = os.path.splitext(video_file.name)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(upload_dir, filename)

    with open(save_path, "wb") as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    try:
        model, device = _get_model()

        if model is None:
            return {
                "error": "Model weights not loaded. "
                         "Please place your .pth file in the model_weights/ folder "
                         "and restart the server."
            }

        num_frames = getattr(settings, "NUM_FRAMES", 20)
        tensor     = load_video_frames(save_path, num_frames=num_frames)
        result     = predict(model, tensor, device)
        result["video_url"] = settings.MEDIA_URL + "uploads/" + filename
        return result

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        return {"error": str(exc)}


# ──────────────────────────────────────────────────────────────────
class IndexView(View):
    template_name = "detector/index.html"

    def get(self, request):
        form = VideoUploadForm()
        return render(request, self.template_name, {"form": form})


# ──────────────────────────────────────────────────────────────────
class PredictView(View):
    template_name = "detector/result.html"

    def post(self, request):
        form = VideoUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(
                request,
                "detector/index.html",
                {"form": form, "errors": form.errors},
            )

        result = _run_inference(request.FILES["video"])
        return render(request, self.template_name, {"result": result})


# ──────────────────────────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class PredictAPIView(View):
    """JSON API: POST a multipart form with field 'video'."""

    def post(self, request):
        if "video" not in request.FILES:
            return JsonResponse({"error": "No video file provided."}, status=400)

        form = VideoUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return JsonResponse({"error": form.errors}, status=400)

        result = _run_inference(request.FILES["video"])
        status_code = 500 if "error" in result else 200
        return JsonResponse(result, status=status_code)
