from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
from torchvision.ops.boxes import batched_nms, box_area
from efficient_sam.build_efficient_sam import build_efficient_sam
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask
    )

def get_efficient_sam_model(mode="s", gpu="cuda:0", checkpoint="/data/checkpoint/efficient_sam_vits.pt"):
    if mode=="s":
        model = build_efficient_sam(encoder_patch_embed_dim=384,
                                    encoder_num_heads=6,
                                    checkpoint=checkpoint).eval()
    else:
        model = build_efficient_sam(encoder_patch_embed_dim=192,
                                   encoder_num_heads=3,
                                   checkpoint=checkpoint).eval()
    return model.to(gpu)

## the EfficientSAM everthing mode
class EfficientSAMEverthing():
    def __init__(self, 
                 grid_size=17, 
                 gpu = "cuda:0",
                 model=None):
        '''
        @params grid_size: the point number of everthing mode 
        @params model: the EfficientSAM model
        '''

        self.grid_size = grid_size
        self.gpu = gpu
        self.model = model.to(gpu)

    ## Get an image format that matches the input by EfficientSAM
    def image_process(self, image_path):
        to_tensor =  transforms.ToTensor()
        sample_image = Image.open(image_path)
        sample_image_tensor = to_tensor(sample_image)
        _, original_image_h, original_image_w = sample_image_tensor.shape
        return sample_image_tensor, (original_image_h, original_image_w)
    
    ## The prompt point for grid_size x grid_size is obtained according to the width and height
    def generate_grid(self, size):
        '''
        @params size: the image shape(height,weight)
        '''
        original_image_h, original_image_w = size
        xy = []
        for i in range(self.grid_size):
            curr_x = 0.5 + i / self.grid_size * original_image_w
            for j in range(self.grid_size):
                curr_y = 0.5 + j / self.grid_size * original_image_h
                xy.append([curr_x, curr_y])
        xy = torch.from_numpy(np.array(xy))
        points = xy
        num_pts = xy.shape[0]
        point_labels = torch.ones(num_pts, 1)
        return points, point_labels, num_pts
    
    ## Mask post-processing strategy, such as the NMS...
    def process_small_region(self, rles, min_area, nms_thresh):
        new_masks = []
        scores = []

        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks
    
    ## Get the minimum bounding box by contour
    def get_bbox_by_contours(self, mask):
        left = top = 800 # the upper left corner
        right = bottom = -1 # the bottom right corner

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        len_contours = len(contours)
        for index in range(len_contours):
            x, y, w, h = cv2.boundingRect(contours[index])
            left = min(left, x)
            top = min(top, y)
            right = max(right, x + w)
            bottom = max(bottom, y + h)
        return (left,top, right, bottom), len_contours


    ## Segment everything, filter small area mask and high coincidence mask
    def segment_everthing(self, image_info=None, image_path=None, min_area=300, nms_thresh = 0.7):
        '''
        @params image_info: the image object and its width and height
        @params image_path: the input image storage path

        Choose between image_info and image_path, image_path is not needed when the image_process returned object has been obtained.
        Otherwise you need an image path
        '''
        assert image_info is not None or image_path is not None , "Neither image_info or image_path should be None!"
        if(image_path is not None):
            image_tensor, size = self.image_process(image_path)
        else:
            image_tensor, size = image_info

        points, point_labels, num_pts = self.generate_grid(size)

        with torch.no_grad():
            predicted_logits, predicted_iou = self.model(
                image_tensor[None, ...].to(self.gpu),
                points.reshape(1, num_pts, 1, 2).to(self.gpu),
                point_labels.reshape(1, num_pts, 1).to(self.gpu),
            )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        predicted_masks = predicted_logits[0]
        iou = predicted_iou[0, :, 0]
        index_iou = iou > 0.7
        iou_ = iou[index_iou]

        masks = predicted_masks[index_iou]
        score = calculate_stability_score(masks, 0.0, 1.0)
        score = score[:, 0]
        index = score > 0.90

        masks = masks[index]
        iou_ = iou_[index]
        masks = torch.ge(masks, 0.0)
        rle = [mask_to_rle_pytorch(m[0:1]) for m in masks]
        rle_masks = self.process_small_region(rle, min_area,nms_thresh)

        out_masks = []
        for mask in rle_masks:
            mask = mask.astype(np.uint8)
            
            contour_area = np.sum(mask)
            if contour_area < min_area: continue
            
            bbox, _ = self.get_bbox_by_contours(mask)
            mask_dict = {}
            mask_dict["segmentation"] = mask
            mask_dict["area"] = contour_area
            mask_dict["bbox"] = bbox
            out_masks.append(mask_dict)
        return out_masks

class EfficientSAMPrompt():
    def __init__(self, gpu="cuda:0", model = None):
        self.gpu = gpu
        self.model = model

    ## Get an image format that matches the input by EfficientSAM
    def image_process(self, image_path):
        to_tensor =  transforms.ToTensor()
        sample_image = Image.open(image_path)
        sample_image_tensor = to_tensor(sample_image)
        _, original_image_h, original_image_w = sample_image_tensor.shape
        return sample_image_tensor, (original_image_h, original_image_w)
    
    ## Segment by point and bounding box prompts
    def segment_prompt(self, box, label, image_info=None, image_path=None):
        assert image_info is not None or image_path is not None , "Neither image_info or image_path should be None!"
        if(image_path is not None):
            image_tensor, _ = self.image_process(image_path)
        else:
            image_tensor, _ = image_info

        pts_sampled = torch.reshape(torch.tensor(box), [1, -1, 2, 2])
        pts_labels = torch.reshape(torch.tensor(label), [1, -1, 2])
        predicted_logits, predicted_iou = self.model(
            image_tensor[None, ...].to(self.gpu),
            pts_sampled.to(self.gpu),
            pts_labels.to(self.gpu),
        )

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        masked = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
        return masked.astype(np.uint8)

if __name__ == "__main__":
    gpu = "cuda:1"
    model = get_efficient_sam_model(gpu=gpu)

    # box_generate = EfficientSAMPrompt(gpu,model)
    # input_point = np.array([[500,200], [750, 550]]) #,  [[x1, y1],[x2, y2]]
    # input_label = np.array([[2,3]])
    # mask = box_generate.segment_prompt(input_point,input_label,image_path="/opt/EfficientSAM/figs/examples/toilet4947.jpg")

    everthing_generate = EfficientSAMEverthing(gpu=gpu, model=model)
    masks = everthing_generate.segment_everthing(image_path="/opt/EfficientSAM/figs/examples/toilet4947.jpg")
    # print(len(masks))
    # for i,mask in enumerate(masks):
    #     print(i,mask["area"])
    #     cv2.imwrite(f"imgs/box{i}.png",np.uint8(mask["segmentation"]*255))









        




