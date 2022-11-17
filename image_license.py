import datetime
import sys

import imutils
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage

import layout_image
import cv2
import numpy as np
import Preprocess
import math

img_path = None
Ivehicle = None

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

#mô hình KNN
npaClassifications = np.loadtxt("classificationS.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)


class MainWindow(QtWidgets.QFrame, layout_image.Ui_Frame):
    def __init__(self,*args, **kwargs):
        super(MainWindow,self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btn_chonanh.clicked.connect(self.loadImage)

        self.btn_nhandang.clicked.connect(self.imgae_license)
        self.btn_info.clicked.connect(self.info)

    def showtime(self):
        while True:
            QApplication.processEvents()
            dt = datetime.datetime.now()
            self.let_ngay.setText('%s-%s-%s' % (dt.day, dt.month, dt.year))
            self.let_gio.setText('%s:%s:%s' % (dt.hour, dt.minute, dt.second))
    def loadImage(self):
        self.img_path = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.Ivehicle = cv2.imread(self.img_path)

        # nguyen goc
        self.cv2_path = cv2.imread(self.img_path)

        self.img_goc = cv2.imread(self.img_path)
        self.setPhoto()

    def setPhoto(self):
        self.Ivehicle = imutils.resize(self.Ivehicle,width=300,height=340)
        frame = cv2.cvtColor(self.Ivehicle, cv2.COLOR_BGR2RGB)
        self.Ivehicle = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.original_img.setPixmap(QtGui.QPixmap.fromImage(self.Ivehicle))


    def imgae_license(self, img_path):

        # Tiền xử lý ảnh
        global first_line, second_line
        imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(self.cv2_path)
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Tách biên bằng canny
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # tăng sharp cho egde (Phép nở)
        # vẽ contour và lọc biển số
        new, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        screenCnt = []
        for c in contours:
            peri = cv2.arcLength(c, True)  # Tính chu vi
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h
            if (len(approx) == 4):
                screenCnt.append(approx)
                cv2.putText(self.cv2_path, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

        if screenCnt is None:
            detected = 0
            print("No plate detected")
        else:
            detected = 1

        if detected == 1:
            for i in screenCnt:
                cv2.drawContours(self.cv2_path, [i], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

                # Tìm góc xoay ảnh
                (x1, y1) = i[0, 0]
                (x2, y2) = i[1, 0]
                (x3, y3) = i[2, 0]
                (x4, y4) = i[3, 0]
                array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                # sắp xếp mảng severse = True -> tăng dần
                sorted_array = array.sort(reverse=True, key=lambda x: x[1])
                (x1, y1) = array[0]
                (x2, y2) = array[1]
                doi = abs(y1 - y2)
                ke = abs(x1 - x2)
                angle = math.atan(doi / ke) * (180.0 / math.pi)

                # Cắt biển số ra khỏi ảnh và xoay ảnh

                mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                new_image = cv2.drawContours(mask, [i], 0, 255, -1, )
                # Now crop
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                roi = self.cv2_path[topx:bottomx, topy:bottomy]
                imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
                ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2
                if x1 < x2:
                    print('x1<x2')
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                else:
                    print('x1>x2')
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
                roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
                imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

                #Tiền xử lý ảnh đề phân đoạn kí tự
                kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
                _, cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #Lọc vùng kí tự
                char_x_ind = {}
                char_x = []
                height, width, _ = roi.shape
                roiarea = height * width

                for ind, cnt in enumerate(cont):
                    (x, y, w, h) = cv2.boundingRect(cont[ind])
                    ratiochar = w / h
                    char_area = w * h

                    if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                        if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                            x = x + 1
                        char_x.append(x)
                        char_x_ind[x] = ind
                char_x = sorted(char_x)
                strFinalString = ""
                first_line = ""
                second_line = ""

                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]  # cắt kí tự ra khỏi hình

                    imgROIResized = cv2.resize(imgROI,
                                               (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize lại hình ảnh
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều

                    # cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
                    npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                            k=3)  # call KNN function find_nearest; neigh_resp là hàng xóm
                    strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

                    if (y < height / 3):  # Biển số 1 hay 2 hàng
                        first_line = first_line + strCurrentChar

                    else:
                        second_line = second_line + strCurrentChar

                print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
                # roi = cv2.resize(roi, None, fx=0.75, fy=0.75)

                self.let_bienso.setText('{}-{}'.format(first_line, second_line))
                # cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                # viết biển số lên anh -> in ra lbl_result
                strFinalString = first_line + second_line
                cv2.putText(self.img_goc,strFinalString, (50, 50),  cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0),
                        lineType=cv2.LINE_AA)
                self.img_goc = imutils.resize(self.img_goc, width=971, height=541)
                frame = cv2.cvtColor(self.img_goc, cv2.COLOR_BGR2RGB)
                self.img_goc = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                self.lbl_result.setPixmap(QtGui.QPixmap.fromImage(self.img_goc))

                break
    def info(self, text):
        in4 = self.let_bienso.text()
        in5 = int(in4[0:2])
        self.let_ten.setText('Hoàng Lê Thiện An')
        lang = {
            11: 'Cao Bằng', 12: 'Lạng Sơn', 14: 'Quảng Ninh', 15: 'Hải Phòng', 17: 'Thái Bình', 18: 'Nam Định',
            19: 'Phú Thọ', 20: 'Thái Nguyên', 21: 'Yên Bái', 22: 'Tuyên Quang', 23: 'Hà Giang', 24: 'Lao Cai',
            25: 'Lai Châu', 26: 'Sơn La', 27: 'Điện Biên', 28: 'Hoà Bình', 29: 'Hà Nội', 30: 'HN', 31: 'Hà Nội',
            32: 'Hà Nội', 33: 'Hà Nội', 40: 'Hà Nội', 34: 'Hải Dương', 35: 'Ninh Bình', 36: 'Thanh Hóa', 37: 'Nghệ An',
            38: 'Hà Tĩnh', 43: 'Đà Nẵng', 47: 'Dak Lak', 48: 'Đắc Nông', 49: 'Lâm Đồng', 50: 'HCM', 51: 'HCM',
            52: 'HCM',
            53: 'HCM', 54: 'HCM', 55: 'HCM', 56: 'HCM', 57: 'HCM', 58: 'HCM', 59: 'HCM', 60: 'Đồng Nai',
            61: 'Bình Dương',
            62: 'Long An', 63: 'Tiền Giang', 64: 'Vĩnh Long', 65: 'Cần Thơ', 66: 'Đồng Tháp', 67: 'An Giang',
            68: 'Kiên Giang',
            69: 'Cà Mau', 70: 'Tây Ninh', 71: 'Bến Tre', 72: 'Vũng Tàu', 73: 'Quảng Bình', 74: 'Quảng Trị', 75: 'Huế',
            76: 'Quảng Ngãi', 77: 'Bình Định', 78: 'Phú Yên', 79: 'Nha Trang', 81: 'Gia Lai', 82: 'Kon Tum',
            83: 'Sóc Trăng',
            84: 'Trà Vinh', 85: 'Ninh Thuận', 86: 'Bình Thuận', 88: 'Vĩnh Phúc', 89: 'Hưng Yên', 90: 'Hà Nam',
            92: 'Quảng Nam',
            93: 'Bình Phước', 94: 'Bạc Liêu', 95: 'Hậu Giang', 97: 'Bắc Cạn', 98: 'Bắc Giang', 99: 'Bắc Ninh',
        }

        for name, code in lang.items():
            if in5 == name:
                self.let_tinh.setText(code)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.show()
    widget.showtime()
    try:
        sys.exit(app.exec_())
    except (SystemError, SystemExit):
        app.exit()

