import cv2
import numpy as np
from knnDetector import knnDetector
from sort import Sort
import time
import adjuster
import tensorflow

def start_detect():
    path = "IMG_4550.MOV"
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 200)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('highThrow-closer.mp4', fourcc, 25, size)

    detector = knnDetector(500, 400, 10)
    cv2.destroyAllWindows()
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("history", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    flag = False

    # 不能忍受漏检，需要预测成功十次才返回预测框，IOU最少0.1
    sort = Sort(3, 5, 0.1)

    ret, frame = capture.read()
    adjust = adjuster.Adjuster(frame, (120, 60))

    index = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame_start = time.time()
        frame = adjust.debouncing(frame)
        print(f"debouncing image take {time.time() - frame_start} s")

        start = time.time()
        mask, bboxs = detector.detectOneFrame(frame)
        print(f"detectOneFrame image take {time.time() - start} s")

        start = time.time()
        if bboxs != []:
            bboxs = np.array(bboxs)
            bboxs[:, 2:4] += bboxs[:, 0:2]
            # test
            # for bbox in bboxs:
            #     cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)
            #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            trackBox = sort.update(bboxs)
        else:
            # test
            trackBox = sort.update()

        print(f"track image take {time.time() - start} s")

        # test
        for bbox in trackBox:
            bbox = [int(bbox[i]) for i in range(5)]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            cv2.putText(frame, str(bbox[4]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # # out.write(frame)
        #
        # cv2.imshow("mask", mask)
        # cv2.imshow("frame", frame)
        if flag:
            if cv2.waitKey(0) == 27:
                flag = False
        else:
            if cv2.waitKey(1) == 27:
                flag = True
        end = time.time()
        print("one frame coast : ", end - frame_start)
        print(index)
        index += 1
    out.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_detect()