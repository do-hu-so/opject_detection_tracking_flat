import datetime
import sys
# pip install pyqt5
import cv2
import imutils
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
import Preprocess
import math
import layout_video
import os


ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

Min_char_area = 0.015
Max_char_area = 0.06

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

max_size_plate = 18000
min_size_plate = 5000

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

tongframe = 0
biensotimthay = 0

#Load KNN model
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)


FJoin = os.path.join
def GetFiles(path):
	file_list, dir_list = [], []
	for dir, subdirs, files in os.walk(path):
		file_list.extend([FJoin(dir, f) for f in files])
		dir_list.extend([FJoin(dir, d) for d in subdirs])
	file_list = filter(lambda x: not os.path.islink(x), file_list)
	dir_list = filter(lambda x: not os.path.islink(x), dir_list)
	return file_list, dir_list

path='D:\Workspace\deep_learning\{}ien_so_xe\LP_Rec_Video-main\{}est.MOV'.format('b','t')
cap = cv2.VideoCapture(path)
# file,dir= GetFiles(path)
# for files in file:
#     cap = cv2.VideoCapture(files) # đầy là đường dẫn

class MainWindow(QtWidgets.QFrame, layout_video.Ui_Frame):
    def __init__(self,*args, **kwargs):
        super(MainWindow,self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btn_chonVideo.clicked.connect(self.start_capture_video)
        self.btn_nhandang.clicked.connect(self.stop_capture_video)
        self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.thread[1].stop()

    def showtime(self):
        while True:
            QApplication.processEvents()
            dt = datetime.datetime.now()
            self.let_ngay.setText('%s-%s-%s' % (dt.day, dt.month, dt.year))
            self.let_gio.setText('%s:%s:%s' % (dt.hour, dt.minute, dt.second))

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
        self.thread[1].updateGui.connect(self.updateBs)
        self.thread[1].signal_roi.connect(self.updateRoi)
        self.thread[1].updateGui.connect(self.info)

    def updateBs(self, text):
        self.let_bienso.setText(text)

    def updateRoi(self,img):
        contour_img = self.convert_cv_qt(img)
        self.lbl_contour.setPixmap(contour_img)

    def info(self, text):
        # in4 = self.let_bienso.text()
        in5 = int(text[0:2])
        self.let_ten.setText('Nguyễn Quang Thái')
        self.let_lop.setText('18IT3 - 18IT120')
        lang = {
            11: 'Cao Bằng', 12: 'Lạng Sơn', 14: 'Quảng Ninh', 15: 'Hải Phòng', 17: 'Thái Bình', 18: 'Nam Định',
            19: 'Phú Thọ', 20: 'Thái Nguyên', 21: 'Yên Bái', 22: 'Tuyên Quang', 23: 'Hà Giang', 24: 'Lao Cai',
            25: 'Lai Châu', 26: 'Sơn La', 27: 'Điện Biên', 28: 'Hoà Bình', 29: 'Hà Nội', 30: 'Hà Nội', 31: 'Hà Nội',
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

    def show_wedcam(self, img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(img)
        self.original_video.setPixmap(qt_img)

    def convert_cv_qt(self, img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1050, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class capture_video(QThread):
    # signal để truyền data
    signal = pyqtSignal(np.ndarray)
    signal_roi = pyqtSignal(np.ndarray)
    updateGui = pyqtSignal(str)

    def __init__(self, index):
        self.index = index

        print("start threading", self.index)
        super(capture_video, self).__init__()

    def run(self):
        # cap = cv2.VideoCapture('test.MOV')
        global tongframe
        while(cap.isOpened()):

            ret, img = cap.read()
            if ret:
                self.signal.emit(img)
            
            tongframe = tongframe + 1
            imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
            canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Tách biên bằng canny
            kernel = np.ones((3, 3), np.uint8)
            dilated_image = cv2.dilate(canny_image, kernel,
                                       iterations=1)  # tăng sharp cho egde (Phép nở). để biên canny chỗ nào bị đứt thì nó liền lại để vẽ contour
            
            # lọc vùng biển số
            contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = []
            

            for c in contours:
                peri = cv2.arcLength(c, True)  # Tính chu vi
                approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
                [x, y, w, h] = cv2.boundingRect(approx.copy())
                ratio = w / h
                if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
                    screenCnt.append(approx)
            if screenCnt is None:
                detected = 0
                print("No plate detected")
            else:
                detected = 1

            if detected == 1:
                n = 1
                for screenCnt in screenCnt:
                    #Tính góc xoay
                    (x1, y1) = screenCnt[0, 0]
                    (x2, y2) = screenCnt[1, 0]
                    (x3, y3) = screenCnt[2, 0]
                    (x4, y4) = screenCnt[3, 0]
                    array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    sorted_array = array.sort(reverse=True, key=lambda x: x[1])
                    (x1, y1) = array[0]
                    (x2, y2) = array[1]

                    doi = abs(y1 - y2)
                    ke = abs(x1 - x2)
                    angle = math.atan(doi / ke) * (180.0 / math.pi)



                    mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

                    # crop
                    (x, y) = np.where(mask == 255)
                    (topx, topy) = (np.min(x), np.min(y))
                    (bottomx, bottomy) = (np.max(x), np.max(y))

                    roi = img[topx:bottomx + 1, topy:bottomy + 1]
                    imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

                    ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

                    if x1 < x2:
                        rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                    else:
                        rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                    roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                    imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

                    roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
                    imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

                    # Tiền xử lý biển số

                    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
                    cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Phân đoạn kí tự
                    char_x_ind = {}
                    char_x = []
                    height, width, _ = roi.shape
                    roiarea = height * width

                    for ind, cnt in enumerate(cont):
                        area = cv2.contourArea(cnt)
                        (x, y, w, h) = cv2.boundingRect(cont[ind])
                        ratiochar = w / h
                        if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                            if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                                x = x + 1
                            char_x.append(x)
                            char_x_ind[x] = ind

                    # Nhận diện kí tự và in ra số xe
                    if len(char_x) in range(7, 10):
                        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                        char_x = sorted(char_x)
                        strFinalString = ""
                        first_line = ""
                        second_line = ""

                        for i in char_x:
                            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]]) # xmin, ymin, xmax-xmin : W, ymax-ymin: H
                            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2) #vẽ hình vuông bo theo giá x,y,w,h
                            # cắt kí tự ra khỏi hình
                            imgROI = thre_mor[y:y + h, x:x + w]

                            imgROIResized = cv2.resize(imgROI, (
                            RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize lại hình ảnh
                            npaROIResized = imgROIResized.reshape(
                                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều
                            # chuyển ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
                            npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
                            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                                    k=3)  # call KNN function find_nearest; neigh_resp là hàng xóm
                            strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
                            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                            if (y < height / 3):  # Biển số 1 hay 2 hàng
                                first_line = first_line + strCurrentChar
                            else:
                                second_line = second_line + strCurrentChar

                        strFinalString = first_line + second_line
                        # print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
                        print("\n bien so la" +" "+ strFinalString + "\n")

                        self.updateGui.emit(strFinalString) #truyền string vào này

                        self.signal_roi.emit(roi)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.show()
    widget.showtime()
    try:
        sys.exit(app.exec_())
    except (SystemError, SystemExit):
        app.exit()

