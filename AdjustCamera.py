import time
import cv2


def adjust_camera(cv2):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    while True:
        _, im = cap.read()
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    adjust_camera(cv2)


if __name__ == '__main__':
    main()