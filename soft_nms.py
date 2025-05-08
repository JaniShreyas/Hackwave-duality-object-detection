import torch
import numpy as np 
from ultralytics.utils.ops import non_max_suppression

def soft_nms(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
    sigma=0.5,  # Gaussian sigma parameter for Soft-NMS
    method='gaussian',  # Methods: 'linear', 'gaussian'
):
    """
    Soft Non-Maximum Suppression for YOLOv8 outputs.
    
    Args:
        prediction (torch.Tensor): predictions from network forward()
        conf_thres (float): confidence threshold
        iou_thres (float): IoU threshold
        classes (list): filter by class
        agnostic (bool): agnostic to NMS
        multi_label (bool): multiple labels per box
        labels (tuple): (cls, score, x1, y1, x2, y2)
        max_det (int): maximum number of detections per image
        nm (int): number of masks
        sigma (float): sigma parameter for gaussian soft-NMS
        method (str): NMS method, 'linear' or 'gaussian'
    
    Returns:
        list of detections, each in (n,6) format, where n is number of detections and 
        6 corresponds to (xyxy, conf, cls)
    """
    # Check if cuda is available
    device = prediction.device
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes put into NMS
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (increases 0.5ms/img)
    merge = False  # use merge-NMS
    
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess
        
        # Batched Soft-NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # Convert to numpy for soft-NMS implementation
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        # Soft-NMS implementation
        indices = soft_nms_cpu(
            boxes_np,
            scores_np,
            iou_thres,
            sigma=sigma,
            thresh=conf_thres,
            method=method
        )
        
        # Keep top max_det indices
        if len(indices) > max_det:
            indices = indices[:max_det]
            
        # Convert indices back to torch tensor
        i = torch.tensor(indices, device=device)
        
        # Final detections
        output[xi] = x[i]
    
    return output

def soft_nms_cpu(dets, scores, iou_threshold, sigma=0.5, thresh=0.001, method='gaussian'):
    """
    Soft NMS implementation on CPU.
    
    Args:
        dets: boxes in format [x1, y1, x2, y2]
        scores: corresponding scores for each box
        iou_threshold: IoU threshold for determining overlap
        sigma: Gaussian parameter if using gaussian method
        thresh: confidence threshold for final detections
        method: 'linear' or 'gaussian'
    
    Returns:
        indices of kept boxes
    """
    # Convert to numpy
    boxes = dets.copy()
    scores_orig = scores.copy()
    
    # Get the indices of boxes sorted by scores (highest first)
    order = scores_orig.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Get IoU of the current box with the rest
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # IoU
        ovr = inter / ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + 
                       (boxes[order[1:], 2] - boxes[order[1:], 0]) * 
                       (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter)
        
        # Apply Soft-NMS based on method
        if method == 'linear':
            # Linear penalty: score = score * (1 - IoU) if IoU > threshold
            inds = np.where(ovr <= iou_threshold)[0]
            scores_orig[order[1:][ovr > iou_threshold]] *= (1 - ovr[ovr > iou_threshold])
        else:  # gaussian
            # Gaussian penalty: score = score * exp(-(IoU^2)/sigma)
            scores_orig[order[1:]] *= np.exp(-(ovr * ovr) / sigma)
            inds = np.where(scores_orig[order[1:]] > thresh)[0]
            
        # Update order and scores
        order = order[inds + 1]
    
    return keep

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# Integration with YOLOv8 - Monkey patching the NMS function
def integrate_soft_nms():
    """
    Replace the standard NMS function with our Soft-NMS implementation
    """
    from ultralytics.utils.ops import non_max_suppression
    import functools
    
    # Store the original function for reference
    original_nms = non_max_suppression
    
    # Replace with soft-NMS with default parameters
    non_max_suppression = functools.partial(
        soft_nms,
        sigma=0.5,
        method='gaussian'
    )
    
    return original_nms  # Return original in case you want to restore it

# Example usage in your project
if __name__ == "__main__":
    # Store the original NMS function if needed later
    original_nms = integrate_soft_nms()
    
    # Now any call to non_max_suppression will use soft_nms instead
    # Your model should now use Soft-NMS without further changes
    
    # To restore original NMS if needed:
    # from ultralytics.utils.ops import non_max_suppression
    # non_max_suppression = original_nms