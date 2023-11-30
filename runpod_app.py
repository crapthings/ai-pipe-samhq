import json
import requests
import numpy as np
from PIL import ImageOps
from diffusers.utils import load_image
from segment_anything_hq import SamPredictor, sam_model_registry
import runpod

from utils import buff_png, upload_image, extract_origin_pathname

sam = sam_model_registry['vit_h'](checkpoint = 'sam_hq_vit_h.pth').to('cuda')
predictor = SamPredictor(sam)

def run (job):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')
        debug = _input.get('debug')

        input_url = _input.get('input_url')
        upload_url = _input.get('upload_url')
        points = _input.get('points')
        # points = json.loads(points)

        # move later
        input_image = load_image(input_url)
        input_image = np.array(input_image)

        predictor.set_image(input_image)

        input_point = np.array(points)
        input_label = np.ones(input_point.shape[0])

        masks, _, _ = predictor.predict(
            point_coords = input_point,
            point_labels = input_label,
        )

        return masks.tolist()
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
