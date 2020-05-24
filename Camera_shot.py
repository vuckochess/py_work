import time
import winsound
import cv2

def make_snapshot(cv2):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    print(cap.get(3), cap.get(4))

    # Make a couple of shots in the air to get brighter pictures!
    for counter in range(8):
        _, im = cap.read()
    # cv2.waitKey(3)
    # cv2.imshow('frame', im)

    counter = 0
    while True:
        winsound.Beep(400, 1000)
        _, im = cap.read()
        time.sleep(2.5)
        file_name = './Snapshots/my_snapshot_' + str(counter+1) + 'a.jpg'
        cv2.imwrite(file_name, im)
        print(file_name)
        _, im = cap.read()
        time.sleep(2.5)
        file_name = './Snapshots/my_snapshot_' + str(counter+1) + 'b.jpg'
        cv2.imwrite(file_name, im)
        print(file_name)
        counter += 1


    cap.release()
    cv2.destroyAllWindows()

def main():
    make_snapshot(cv2)


if __name__ == '__main__':
    main()