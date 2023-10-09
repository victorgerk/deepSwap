import threading
from functools import lru_cache

import numpy
import opennsfw2
from PIL import Image
from keras import Model

from facefusion.typing import Frame

PREDICTOR = None
THREAD_LOCK : threading.Lock = threading.Lock()
MAX_PROBABILITY = 0.75
FRAME_INTERVAL = 25
STREAM_COUNTER = 0


def get_predictor() -> Model:
	global PREDICTOR

	with THREAD_LOCK:
		if PREDICTOR is None:
			PREDICTOR = opennsfw2.make_open_nsfw_model()
	return PREDICTOR


def clear_predictor() -> None:
	global PREDICTOR

	PREDICTOR = None


def predict_stream(frame : Frame) -> bool:
	return False


def predict_frame(frame : Frame) -> bool:
	return false


@lru_cache(maxsize = None)
	return false


@lru_cache(maxsize = None)
def predict_video(video_path : str) -> bool:
	return false
