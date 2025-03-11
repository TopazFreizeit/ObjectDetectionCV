import cv2
import torch
import yaml
import os
import argparse
import torchvision.transforms.v2 as v2
from model.ssd import SSD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(config_path, device):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    model = SSD(config=model_config, num_classes=dataset_config['num_classes'])
    model = model.to(device)
    model.eval()

    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint exists at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    return model, config

def preprocess_frame(frame, im_size, mean, std):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
    transform = v2.Compose([
        v2.Resize((im_size, im_size)),
        v2.Normalize(mean=mean, std=std)
    ])
    frame_tensor = transform(frame_tensor)
    return frame_tensor.unsqueeze(0).to(device)

def draw_detections(frame, detections, frame_w, frame_h, label_name='person'):
    if len(detections) == 0 or 'boxes' not in detections[0]:
        return
    
    boxes = detections[0]['boxes'].detach().cpu().numpy()
    labels = detections[0]['labels'].detach().cpu().numpy()
    scores = detections[0]['scores'].detach().cpu().numpy()

    max_score = -1
    max_score_idx = -1
    
    for i, (label, score) in enumerate(zip(labels, scores)):
        if label == 1 and score > max_score:
            max_score = score
            max_score_idx = i
    
    if max_score_idx >= 0:
        box = boxes[max_score_idx]
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * frame_w), int(y1 * frame_h), int(x2 * frame_w), int(y2 * frame_h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        text = f"{label_name}: {max_score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def process_video(video_path, output_path, model, config):
    model.low_score_threshold = config['train_params']['infer_conf_threshold']
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    im_size = config['dataset_params']['im_size']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = preprocess_frame(frame, im_size, mean, std)
        
        with torch.no_grad():
            _, detections = model(frame_tensor)

        draw_detections(frame, detections, frame_width, frame_height)

        out.write(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print("Video processing completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection on video using SSD')
    parser.add_argument('--config', default='config/voc.yaml', type=str, help='Path to config file')
    parser.add_argument('--video', default='input_video.mp4', type=str, help='Path to input video file')
    parser.add_argument('--output', default='output_video.mp4', type=str, help='Path to output video file')
    args = parser.parse_args()

    model, config = load_model(args.config, device)
    
    process_video(args.video, args.output, model, config)