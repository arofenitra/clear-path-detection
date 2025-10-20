from ultralytics import YOLO
import supervision as sv
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("../model/yolov8n.pt").to(device)
import imageio
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
def object_detection_image(model, image):
    # Supervision annotators  
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.5,text_thickness=1)
        
    results = model(image, conf=0.25, iou=0.45,verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections)==0:
        return image.copy()
    annotated_image = image.copy()
    image = box_annotator.annotate(scene=annotated_image, detections = detections)
    image = label_annotator.annotate(scene=annotated_image,detections=detections)
    return image
def segmentation_image(model,image):
    mask_annotator = sv.MaskAnnotator(opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_scale=0.5,text_thickness=1)
        
    results = model(image,task="segment", conf=0.25, iou=0.45,verbose=False,device="cpu")[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections)==0:
        return image.copy()
    annotated_image = image.copy()
    image = mask_annotator.annotate(scene=annotated_image, detections = detections)
    image = label_annotator.annotate(scene=annotated_image,detections=detections)
    return image
def resize_and_pad(image, target_size=(640,640)):
    h,w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(image,(new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding the image 
    # pad_w, pad_h = target_size[0]-new_w, target_size[1]-new_h
    # top = pad_h//2
    # bottom = pad_h - top
    # left = pad_w//2
    # right = pad_w - left
    # padded = cv2.copyMakeBorder(resized,top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    # return padded
    return resized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F


def depth_estimation_image(model, transform, image_bgr, device):
    # image_bgr: np.ndarray in BGR (cv2)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Transform → ensure batch dim
    inp = transform(img_rgb).to(device)
    if inp.ndim == 3:  # (C, H, W)
        inp = inp.unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        pred = model(inp)
        # Ensure shape is (N=1, C=1, H, W) for interpolate
        if pred.ndim == 4 and pred.shape[1] == 1:
            pass
        elif pred.ndim == 4 and pred.shape[1] != 1:
            pred = pred[:, :1, ...]  # take first channel
        elif pred.ndim == 3:
            pred = pred.unsqueeze(1)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected MiDaS output shape: {tuple(pred.shape)}")

        # Resize to original image size → (H, W)
        h, w = img_rgb.shape[:2]
        pred = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False)
        depth = pred.squeeze(0).squeeze(0).cpu().numpy()

    # Normalize for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    return depth_color, depth_norm
def clear_pixel_path(image_bgr, model_obj_seg, model_depth, midas_transform, device,
                     group_width=64, focus_band=(0.7, 1.0),
                     occ_max=0.02, near_threshold=0.6, near_max=0.2,
                     return_scores=False):
    """
    image_bgr: np.ndarray (BGR)
    model_obj_seg: ultralytics.YOLO (seg model)
    model_depth: MiDaS model
    midas_transform: MiDaS transform
    device: torch.device
    group_width: 64 → 10 groups across 640 width
    focus_band: (start_h_frac, end_h_frac), part of image height to evaluate (e.g., road band)
    occ_max: max fraction of pixels occupied by objects in the region
    near_threshold: depth_norm > threshold considered 'near'
    near_max: max fraction of 'near' pixels allowed
    return_scores: if True returns per-group metrics

    Returns:
      - clear_groups: list of group indices that are clear
      - ranges: list of (x0, x1) pixel ranges in 640-scale image
      - (optional) scores: list of dict per group with occ_ratio, near_ratio
    """
    # Ensure 640x640 input
    # img_640 = cv2.resize(image_bgr, (640, 640), interpolation=cv2.INTER_AREA)
    # img_640 = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    img_640 = image_bgr
    h, w = img_640.shape[:2]
    # assert w == 640 and h == 640, "Expected 640x640 after resize"

    # Segmentation → combined object mask
    results = model_obj_seg(img_640, task="segment", conf=0.25, iou=0.45, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections)==0:
        a = img_640.copy()
        # y0 = int(h * focus_band[0]); y1 = int(h * focus_band[1])
        y0 = int(h * 0.95); y1 = int(h * 1.0)
        x0,x1 = 0,w
        color = (0, 220, 0) 
        cv2.rectangle(a, (x0, y0), (x1, y1), color, -1)
        return a
    palette = sv.ColorPalette.DEFAULT
    mask_annotator = sv.MaskAnnotator(opacity=0.5,color=palette)
    label_annotator = sv.LabelAnnotator(text_scale=0.5,text_thickness=1,color=palette)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
    annotated_image = img_640.copy()
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections = detections)
    annotated_image = box_annotator.annotate(scene=annotated_image,detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image,detections=detections)
    
    if hasattr(detections, "mask") and detections.mask is not None and len(detections) > 0:
        # detections.mask: (N, H, W) boolean/uint8
        masks = detections.mask.astype(bool)
        combined_mask = masks.any(axis=0)
    else:
        combined_mask = np.zeros((h, w), dtype=bool)

    # Depth (normalized 0..1, higher = nearer per your note)
    _, depth_norm = depth_estimation_image(model_depth, midas_transform, img_640, device)

    # Focus band (e.g., lower 30% of image)
    y0 = int(h * focus_band[0])
    y1 = int(h * focus_band[1])
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if y1 <= y0:
        y0, y1 = 0, h

    num_groups = w // group_width
    clear_groups = []
    ranges = []
    scores = []

    for i in range(num_groups):
        x0 = i * group_width
        x1 = x0 + group_width

        occ_roi = combined_mask[y0:y1, x0:x1]
        depth_roi = depth_norm[y0:y1, x0:x1]

        # Ratios
        occ_ratio = float(occ_roi.mean()) if occ_roi.size else 0.0
        near_ratio = float((depth_roi > near_threshold).mean()) if depth_roi.size else 0.0

        is_clear = (occ_ratio <= occ_max) and (near_ratio <= near_max)
        if is_clear:
            clear_groups.append(i)
            ranges.append((x0, x1))

        if return_scores:
            scores.append({"group": i, "x_range": (x0, x1),
                           "occ_ratio": occ_ratio, "near_ratio": near_ratio,
                           "clear": is_clear})

    if return_scores:
        return annotated_image,clear_groups, ranges, scores
    return annotated_image, clear_groups, ranges
def visualize_groups(img_bgr_640, clear_groups, group_width=64, focus_band=(0.7, 1.0)):
    img_viz = img_bgr_640.copy()
    h, w = img_viz.shape[:2]
    # y0 = int(h * focus_band[0]); y1 = int(h * focus_band[1])
    y0 = int(h * 0.95); y1 = int(h * 1.0)
    for i in range(w // group_width):
        # x0 = i * group_width; x1 = x0 + group_width
        # color = (0, 220, 0) if i in clear_groups else (0, 0, 220)
        # cv2.rectangle(img_viz, (x0, y0), (x1, y1), color, 2)
        if i in clear_groups:
            x0 = i * group_width; x1 = x0 + group_width
            color = (0, 220, 0) 
            cv2.rectangle(img_viz, (x0, y0), (x1, y1), color, -1)
    return img_viz

# import time
# # midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
# midas_transform = midas_transforms.small_transform
# print("Loaded MiDaS_small on", device)
# model = YOLO(r"C:\Users\arofe\Aro_Data\IW_cv\Model\yolov8n-seg.pt")
# image_path = r"C:\Users\arofe\Aro_Data\IW_cv\images\image4.jpg"

import requests 
import cv2 
import supervision as sv 
import torch 
from ultralytics import YOLO 
import traceback
IP = "http://10.16.110.97:9191"
IP_WEBCAM_URL = f"{IP}/shot.jpg" # JPEG mode
VIDEO_STREAM_URL =f"{IP}/video" # MPEG Mode
def connect_ip_webcam_jpg(url):
    try:
        response = requests.get(url,timeout=5)
        if response.status_code ==200:
            img_array = np.array(bytearray(response.content),dtype=np.uint8)
            frame = cv2.imdecode(img_array,-1) # -1 means to keep the original format
            return frame 
        return None 
    except Exception :
        traceback.print_exc()
        return None 
def connect_ip_webcame_stream(url):
    # Using video stream 
    cap = cv2.VideoCapture(url)
    # Set buffer size to minimum for real time processing 
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    return cap 
model_path="../model/yolov8n-seg.pt",
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 

midas= torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
model_seg = YOLO("yolov8n-seg.pt").to(device)      

import time
import cv2

class TimeBasedProcessor:
    def __init__(self, camera_url, processing_interval=0.3):  # 0.3 seconds
        self.camera_url = camera_url
        self.processing_interval = processing_interval  # Process every 0.3 seconds
        self.last_processing_time = 0
        self.cap = self.create_optimized_capture()
        
    def create_optimized_capture(self):
        cap = cv2.VideoCapture(self.camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    
    def should_process_frame(self):
        """Check if enough time has passed to process next frame"""
        current_time = time.time()
        if current_time - self.last_processing_time >= self.processing_interval:
            self.last_processing_time = current_time
            return True
        return False
    
    def run(self):
        frame_count = 0
        processed_count = 0
        
        while True:
            # Always read frames to keep stream alive and clear buffer
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Only process if time interval has passed
            if self.should_process_frame():
                processed_count += 1
                
                # Your processing code here
                processed_frame = self.your_processing_function(frame)
                
                # Display or use the result
                cv2.imshow('Throttled Processing', processed_frame)
                print(f"Frames: {frame_count} | Processed: {processed_count} | Ratio: {processed_count/frame_count:.2f}")
            
            # Always display the latest frame (for smooth video)
            else:
                cv2.imshow('Throttled Processing', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

import time
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch

class ThrottledIPWebcamProcessor:
    def __init__(self, camera_url, model_seg=model_seg,midas=midas,midas_transform=midas_transform, device=device, process_interval=1):
        self.camera_url = camera_url
        self.device = device
        self.process_interval = process_interval  # Seconds between processing
        self.last_process_time = 0
        self.model_seg = model_seg
        self.midas = midas 
        self.midas_transform = midas_transform 
        
        # Initialize camera with minimal buffer
        self.cap = cv2.VideoCapture(camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # High capture FPS
        
        # Initialize models
        # self.model_seg = YOLO(model_path).to(device)
        # self.midas, self.midas_transform = self.setup_depth_model(device)
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.latest_result = None
        
    def setup_depth_model(self, device):
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
        return model.to(device).eval(), transform
    
    def should_process(self):
        """Determine if current frame should be processed based on time"""
        current_time = time.time()
        if current_time - self.last_process_time >= self.process_interval:
            self.last_process_time = current_time
            return True
        return False
    
    def clear_buffer(self):
        """Clear any buffered frames to get the latest one"""
        for _ in range(2):  # Grab 2 frames to clear buffer
            self.cap.grab()
    
    def process_frame(self, frame):
        """Your existing processing function"""
        # Resize for consistent processing
        frame_resized = cv2.resize(frame, (640, 640))
        
        # Your processing pipeline
        A = clear_pixel_path(
            frame_resized, self.model_seg, self.midas, self.midas_transform, self.device,
            group_width=64, focus_band=(0.6, 1.0),
            occ_max=0.02, near_threshold=0.6, near_max=0.2,
            return_scores=False
        )
        if len(A)>4:
            annotated_image = A
            clear_groups = []
        else:

            annotated_image, clear_groups, ranges = A
        
        viz_frame = visualize_groups(annotated_image, clear_groups, 
                                   group_width=64, focus_band=(0.6, 1.0))
        
        return viz_frame, clear_groups
    
    def run(self):
        """Main processing loop with time-based throttling"""
        print(f"Starting processing with {self.process_interval}s interval...")
        
        while True:
            # Clear buffer to get latest frame
            self.clear_buffer()
            
            # Read current frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
            
            self.total_frames += 1
            
            # Check if we should process this frame based on time
            if self.should_process():
                self.processed_frames += 1
                
                try:
                    # Process the frame (this takes time)
                    processed_frame, clear_groups = self.process_frame(frame)
                    self.latest_result = processed_frame
                    
                    # Display processing result
                    cv2.imshow('Clear Path Detection', processed_frame)
                    
                    # Print statistics
                    ratio = self.processed_frames / self.total_frames
                    print(f"Total: {self.total_frames} | Processed: {self.processed_frames} | Ratio: {ratio:.2f}")
                    if clear_groups:
                        print(f"Clear paths: {clear_groups}")
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    # Fallback: display original frame
                    cv2.imshow('Clear Path Detection', frame)
            
            else:
                # Display the latest processed result or current frame
                if self.latest_result is not None:
                    cv2.imshow('Clear Path Detection', self.latest_result)
                else:
                    cv2.imshow('Clear Path Detection', frame)
            
            # Calculate actual FPS
            current_time = time.time()
            actual_fps = 1.0 / (current_time - self.last_process_time) if ((self.processed_frames > 0)and (current_time - self.last_process_time!=0)) else 0
            
            # Display FPS info on frame
            display_frame = cv2.imshow('Clear Path Detection', 
                                     self.latest_result if self.latest_result is not None else frame)
            
            # Add text overlay
            if display_frame is not None:
                cv2.putText(display_frame, f"FPS: {actual_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Processed: {self.processed_frames}/{self.total_frames}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Final stats: Processed {self.processed_frames} of {self.total_frames} frames")

# Usage
if __name__ == "__main__":
    processor = ThrottledIPWebcamProcessor(
        camera_url=f"{IP}/video",
        # model_path=r"C:\Users\arofe\Aro_Data\IW_cv\Model\yolov8n-seg.pt",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        process_interval=0.9  # Process every 0.3 seconds
    )
    processor.run()

