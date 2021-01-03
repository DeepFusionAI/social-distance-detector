from social_distancing_config import HEIGHT, WIDTH
import numpy as np


""" Utilities for pedestrian detection """

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor



""" Pedestrian detector """

def detector(interpreter, image, threshold):
    
    """Returns a list of detection results, each as a tuple of object info."""
    H,W=HEIGHT,WIDTH
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    """ Get all output details """
    boxes = get_output_tensor(interpreter, 0)
    class_id = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    results = []
    for i in range(count):
        if class_id[i] == 0 and scores[i] >= threshold:
            [ymin,xmin,ymax,xmax]=boxes[i]
            (left, right, top, bottom) = (int(xmin * W), int(xmax * W), int(ymin * H), int(ymax * H))
            area=(right-left+1)*(bottom-top+1)
            if area>=1500:
                continue
            centerX=left+int((right-left)/2)
            centerY=top+int((bottom-top)/2)
            results.append((scores[i],(left,top,right,bottom),(centerX,centerY)))
    return results