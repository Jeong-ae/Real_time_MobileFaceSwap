from flask import Flask, Response, render_template
import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

app = Flask(__name__)

def get_id_emb(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

def generate_frames(source_img_path):
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else 'cpu')
    faceswap_model = FaceSwap(use_gpu=paddle.is_compiled_with_cuda())

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')
    

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    
    id_img = cv2.imread(source_img_path)
    landmark = landmarkModel.get(id_img)
    if landmark is None:
        yield (b'No face detected; please adjust camera.')
        return
    aligned_id_img, _ = align_img(id_img, landmark)
    id_emb, id_feature = get_id_emb(id_net, aligned_id_img)
    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        original_frame = frame.copy()

        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = cv2paddle(att_img)
            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            frame = dealign(res, frame, back_matrix, mask)
        else:
            print('**** No Face Detected ****')

        combined_frame = cv2.hconcat([original_frame, frame])
        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    source_img_path = '/user/target3.JPG'  # Make sure to update this path
    return Response(generate_frames(source_img_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Return a main page with the video stream embedded
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
