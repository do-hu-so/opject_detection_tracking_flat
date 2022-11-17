import sys
import numpy as np
import cv2

MIN_CONTOUR_AREA = 40

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    # đọc ảnh train
    imgTrainingNumbers = cv2.imread("train.png")
    # chuyển sang ảnh xám
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    #Thresold
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input Img Blur
                                      255, # vượt ngưỡng trắng
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)

    cv2.imshow("imgThresh", imgThresh)
    imgThreshCopy = imgThresh.copy()

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              cv2.RETR_EXTERNAL, # lấy ra đường phía ngoài
                                                              cv2.CHAIN_APPROX_SIMPLE)

    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []

    # mã ascii
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:
        # nếu đường viền > cỡ cho trước
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # vẽ HCN
            cv2.rectangle(imgTrainingNumbers,
                          (intX, intY),
                          (intX + intW, intY + intH),
                          (0, 0, 255),
                          2)

            # cắt kí tự
            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)

            cv2.imshow("training_numbers.png",
                       imgTrainingNumbers)

            intChar = cv2.waitKey(0)

            if intChar == 27: # esc -> exit
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(
                    intChar)
                # Là file chứa label của tất cả các ảnh mẫu, tổng cộng có 32 x 5 = 160 mẫu.
                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # làm phẳng ảnh

                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0) # thêm ảnh phẳng vào danh sách

    fltClassifications = np.array(intClassifications,
                                  np.float32)
    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))

    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)  # ghi ảnh phẳng vào tệp
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()





