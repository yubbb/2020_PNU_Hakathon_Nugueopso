import requests
from PIL import ImageGrab
import cv2, time
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from imageai.Detection import ObjectDetection

PATH_MODEL = './model/'

# 오브젝트 디텍션 모델 불러오기
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(PATH_MODEL + 'nugueopso_object_detection.h5'))
detector.loadModel()

# 딥러닝 모델 불러오기
model = load_model(PATH_MODEL + 'nugueopsu_deep_learning.h5')

# 이미지 보여주기
def show(img):
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def photo_cut(p,n,h,w):
    ratio = 2.5
    if n==1:
        return p[:int(h/ratio),:int(w/ratio)]
    elif n==2:
        return p[:int(h/ratio),int(w/ratio):]
    elif n==3:
        return p[int(h/ratio):,:int(w/ratio)]
    elif n==4:
        return p[int(h/ratio):,int(w/ratio):]
    elif n==5:
        return p[:,:int(w/ratio)]
    elif n==6:
        return p[:,int(w/ratio):]
    else:
        return p


def screen_shot(view_location):
    time.sleep(3)
    img_ori = np.array(ImageGrab.grab())
    bg_w = img_ori.shape[1]
    ori_w = 1280
    bg_ratio = bg_w / ori_w
    height, width, channel = img_ori.shape
    img_ori = photo_cut(img_ori, view_location, height, width)

    test = img_ori.copy()
    test_val = test[0][0].copy()
    test_val[0] = 0
    test_val[1] = 0
    test_val[2] = 0

    val_list = []
    for i in range(20, 27):
        v = test_val.copy()
        v[0] = i
        v[1] = i
        v[2] = i
        val_list.append(v)

    for i in range(len(val_list)):
        test = np.where(test[:, :] == val_list[i], test_val, test[:, :])

    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(test)
    temp = fgmask.copy()

    r = 150
    conti_list = [[j for j in range(len(temp[:, i]) - r) if sum(temp[:, i][j:j + r]) == 255 * r] for i in
                  range(temp.shape[1])]

    for i in range(len(conti_list) - 2):
        if len(conti_list[i]) != 0 and len(conti_list[i + 1]) != 0 and len(conti_list[i + 2]) != 0:
            break
    st = i

    for j in range(i, len(conti_list)):
        if len(conti_list[j]) == 0:
            break
    ed = j

    a = conti_list[st]
    check_list = [[a[i], a[i + 1]] for i in range(len(a) - 1) if a[i] + 1 != a[i + 1]]
    for i in range(len(check_list)):
        temp[:, st][check_list[i][0] + r:check_list[i][1]] = 255

    a = conti_list[ed - 1]
    check_list = [[a[i], a[i + 1]] for i in range(len(a) - 1) if a[i] + 1 != a[i + 1]]
    for i in range(len(check_list)):
        temp[:, ed][check_list[0][0] + r:check_list[0][1]] = 255

    img_thresh = cv2.adaptiveThreshold(
        temp,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=10
    )

    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    temp_result = img_ori.copy()
    temp_result = np.where(temp_result[:, :, :] == 0, temp_result[:, :, :], 0)
    cv2.drawContours(
        temp_result,
        contours=contours,
        contourIdx=-1,
        color=(255, 255, 255)
    )

    temp_result = img_ori.copy()
    temp_result = np.where(temp_result[:, :, :] == 0, temp_result[:, :, :], 0)
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
        })

    # 영역의 최소 너비, 높이
    MIN_WIDTH, MIN_HEIGHT = 128 * bg_ratio, 72 * bg_ratio
    # 영역의 최소 넓이
    MIN_AREA = MIN_WIDTH * MIN_HEIGHT
    RATIO = 128 / 72
    RATIO_ERROR = 0.2
    MIN_RATIO, MAX_RATIO = 1 - RATIO_ERROR, RATIO * (1 + RATIO_ERROR)
    possible_contours = []

    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            possible_contours.append(d)

    temp_result2 = img_ori.copy()
    temp_result2 = np.where(temp_result2[:, :, :] == 0, temp_result2[:, :, :], 0)
    for d in possible_contours:
        cv2.rectangle(temp_result2,
                      pt1=(d['x'], d['y']),
                      pt2=(d['x'] + d['w'], d['y'] + d['h']),
                      color=(255, 255, 255),
                      thickness=2
                      )

    res = []
    for d in possible_contours:
        x = d['x']
        y = d['y']
        w = d['w']
        h = d['h']
        res.append(abs(d['w'] / d['h'] - width / height))

    for i in range(len(res)):
        cont = possible_contours[res.index(sorted(res)[i])]
        x = cont['x']
        y = cont['y']
        w = cont['w']
        h = cont['h']

        final_img = cv2.cvtColor(img_ori[y:y + h, x:x + w], cv2.COLOR_RGB2BGR)

        print()
        text_print('6_capture_yes_or_no')
        cv2.imshow("test", final_img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 121:
            return final_img, x, y, w, h, view_location, height, width
    time.sleep(1)
    return []

def od_person(detection):
    return [detection[det] for det in range(len(detection)) if detection[det]['name']=='person']


def overlap_area(a, b):
    x_left_up = max(a[0], b[0])
    x_right_up = min(a[2], b[2])

    y_left_up = max(a[1], b[1])
    y_left_down = min(a[3], b[3])

    if x_left_up < x_right_up and y_left_up < y_left_down:
        return (x_right_up - x_left_up) * (y_left_down - y_left_up)
    else:
        return 0

def area(a):
    return (a[2]-a[0])*(a[3]-a[1])

def overlap_remove(temp,per):
    locat = [tuple(temp[i]['box_points']) for i in range(len(temp))]
    overlap_locat = list(set([tuple(sorted([tuple(i),tuple(j)])) for i in locat for j in locat if min(area(i),area(j))*per<overlap_area(i,j) if i!=j]))
    unique_locat = [i for i in locat if i not in [i[0] for i in overlap_locat] + [i[1] for i in overlap_locat]]
    for i in range(len(overlap_locat)):
        ov_ratio = [overlap_area(overlap_locat[i][0],overlap_locat[i][1])/area(ov_loc) for ov_loc in overlap_locat[i]]
        overlap_locat[i] = overlap_locat[i][ov_ratio.index(max(ov_ratio))]
    return [temp[locat.index(i)] for i in list(set(overlap_locat))+unique_locat]

def detect_pop(temp,val,pp,npp):
    if len(temp)<2:
        return temp
    temp = [i for i in temp if i['percentage_probability']>npp]
    while len(temp)!=val:
        pot_list = [i['box_points'] for i in temp]
        per_list = [i['percentage_probability'] for i in temp]
        max_list = [max([overlap_area(i,j)/area(j) for i in pot_list if i!=j]) for j in pot_list]
        pop_list = max_list.copy()
        max_num = [pop_list[i] for i in range(len(pop_list)) if per_list[i]<pp]
        if len(max_num)==0:
            max_num = [pop_list[i] for i in range(len(pop_list))]
        pop_index = max_list.index(pop_list.pop(pop_list.index(max(max_num))))
        max_list.pop(pop_index)
        per_list.pop(pop_index)
        temp.pop(pop_index)
    return temp

def text_print(t):
    f = open("./text/"+t+".txt",'r')
    r = f.read()
    print(r)
    f.close()

normal_area_per = 0.954       # 기본 영역 제거 비율
equal_area_per = 0.91         # model=od 일때 영역 제거 비율
od_big_person_per = 68.07     # od값이 더 클때 얼마 이상을 사람으로 인정 해줄지
od_big_no_person_per = 51     # od값이 더 클때 얼마 이하를 사람으로 인정 안할지

print()
text_print('1_intro')

while True:
    print()
    text_print('2_capture')
    view_location = input("입력 : ")
    try:
        view_location = int(view_location)
    except:
        print()
        text_print('3_error')
    else:
        if view_location<1 or view_location>7:
            print()
            text_print('3_error')
        else:
            break

def pred(pred_val):
    res = [int(round(i,0)) for i in pred_val[0]]
    return res.index(max(res))

dir_name = "nugueopso"
if dir_name not in os.listdir(".."):
    os.makedirs("../"+dir_name)

if view_location!=7:
    while True:
        photo = screen_shot(view_location)

        if len(photo) != 0:
            print()
            text_print('4_capture_yes')
            print()
            break
        else:
            print()
            text_print('5_capture_no')
            print()

    x,y,w,h,view_location,height,width = [photo[i+1] for i in range(7)]
#
P = 0.25
img_w,img_h = int(1280*P),int(720*P)

while True:
    try:
        f = open("./text/7_save_yes_or_no.txt",'r')
        re = f.read()
        f.close()
        save_tf = input(re)
    except:
        print()
        text_print('3_error')
    else:
        if save_tf!='y' or save_tf!='n':
            break
        else:
            print()
            text_print('3_error')

pr_cnt = 1

while True:
    # 이미지 가져오기
    if view_location != 7:
        img = cv2.cvtColor(photo_cut(np.array(ImageGrab.grab()),view_location,height,width),cv2.COLOR_RGB2BGR)[y:y+h,x:x+w]
    else:
        img = cv2.cvtColor(np.array(ImageGrab.grab()),cv2.COLOR_RGB2BGR)
    img1 = cv2.resize(img,(img_w,img_h))

    # 딥러닝 예측 array 생성
    img1 = image.img_to_array(img1)  # 이미 array 형태로 들어온다. 그리고 값이 동일하다.
    img1 = np.expand_dims(img1, axis=0)

    # Object Detection 에서 img에 있는 사람 좌표 파악
    detections = detector.detectObjectsFromImage(input_image=img,
                                                 input_type='array',
                                                 minimum_percentage_probability=50,
                                                 output_type='array')
    detect = od_person(detections[1])
    detect = overlap_remove(detect, normal_area_per)

    # 딥러닝 모델에서 img에 있는 사람 수 파악
    model_pred = pred(model.predict(img1))

    if len(detect) == model_pred:
        detect = overlap_remove(detect, equal_area_per)

    if model_pred < len(detect):
        detect = detect_pop(detect, model_pred, od_big_person_per, od_big_no_person_per)

    for j in range(len(detect)):
        cv2.rectangle(img, (detect[j]['box_points'][0], detect[j]['box_points'][1]),
                      (detect[j]['box_points'][2], detect[j]['box_points'][3]), (0, 0, 255), 3)

    m = {"id": "3504", "current": str(len(detect))}
    res = requests.post("http://3.34.52.3/store/seat/put", data=m)

    if pr_cnt==1:
        print()
        text_print('8_start')
        pr_cnt = 0

    print("\r",len(detect), end='') # 관측 인원수

    if save_tf=='y':
        cv2.imwrite("../"+dir_name+"/"+dir_name+'_'+str(len(os.listdir("../"+dir_name)))+".jpg",img)