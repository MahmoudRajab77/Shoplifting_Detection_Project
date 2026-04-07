"""
Video upload form with extension validation.
"""

from django import forms


ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label="Upload Video",
        help_text=f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}",
        widget=forms.ClearableFileInput(attrs={
            "accept": ",".join(ALLOWED_EXTENSIONS),
            "id": "video-input",
        }),
    )

    def clean_video(self):
        video = self.cleaned_data["video"]
        name  = video.name.lower()
        if not any(name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise forms.ValidationError(
                f"Unsupported file type. Please upload one of: "
                f"{', '.join(ALLOWED_EXTENSIONS)}"
            )
        return video
