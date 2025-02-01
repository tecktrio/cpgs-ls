# your_app/routing.py

from django.urls import re_path
from . import consumers


websocket_urlpatterns = [
    # re_path(r'auto_coordinate_finder', consumers.AutoCoordinateFinder.as_asgi()),  # Update the path as needed
    # re_path(r'manual_coordinate_finder', consumers.ManualCoordinateFinder.as_asgi()),  # Update the path as needed
    re_path(r'', consumers.ServerConsumer.as_asgi()),  # Update the path as needed
]
