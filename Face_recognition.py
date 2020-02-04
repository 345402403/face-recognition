# -*- coding: utf-8 -*-
import cv2
import sys
from train import Model


def direction(x, y, w, h):
    sw = 1200
    sh = 800
    balenceW = 350
    balenceH = 350
    msg = ''
    if w <= balenceW + 30 and w > balenceW - 30:
        msg = 'HOLD'
    elif w > balenceW + 30:
        msg = 'BACK'
    elif w < balenceW - 30:
        msg = 'AHEAD'

    centerX = w / 2 + x
    centerY = h / 2 + y
    screenCenterX = sw / 2
    screenCenterY = sh / 2

    msg = msg + ','

    if centerX <= screenCenterX:
        msg = msg + 'LEFT'
    elif centerX > screenCenterX:
        msg = msg + 'RIGHT'
    return msg

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    model.load_model(file_path='/Users/limeng/Pictures/opencv/lm_face_model.h5')

    # 框住人脸的矩形边框颜色
    color = (255, 255, 255)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "/Users/limeng/code/python/Face_Recog/haarcascade_frontalface_default.xml"
    # 使用人脸识别分类器，读入分类器
    cascade = cv2.CascadeClassifier(cascade_path)
    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        frame = cv2.flip(frame, 1, dst=None)  # 水平镜像

        if ret is True:
            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID, prob = model.face_predict(image)
                print("faceID", faceID)
                # 如果是“我”

                tag = 'alias'
                if faceID == 0 and prob[0][0] > 0.5:
                    tag = 'limeng:' + str(prob[0][0])[0:5]
                elif faceID == 1 and prob[0][1] > 0.5:
                    tag = 'wjn:' + str(prob[0][1])[0:5]
                elif faceID == 2 and prob[0][2] > 0.5:
                    tag = 'mom:' + str(prob[0][2])[0:5]
                elif faceID == 3 and prob[0][3] > 0.5:
                    tag = 'pp:' + str(prob[0][3])[0:5]
                else:
                    tag = 'alias'

                cv2.rectangle(frame, (x - 8, y - 8), (x + w + 8, y + h + 8), color, thickness=1)
                cv2.putText(frame, str(tag),
                            (x + 30, y - 30),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            0.8,  # 字号
                            (20, 0, 200),  # 颜色
                            2)  # 字的线宽
                cv2.putText(frame, direction(x,y,w,h),
                            (20, 20),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            0.8,  # 字号
                            (20, 0, 200),  # 颜色
                            2)  # 字的线宽
                pass
        cv2.imshow("Face Recognition", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

