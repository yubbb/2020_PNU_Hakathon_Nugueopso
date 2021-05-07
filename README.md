# 2020_PNU_Hakathon_Nugueopso
Deep Learning Model, Screen Capture EXE, and Final Results of Hackathon "Nugueopso" Team organized by Pusan National University in 2020

### 개요

1. 대회 명 : 2020년 부산대학교 창의 융합 해커톤

2. 주제 : 새로운 SW & service 개발

3. 기간 : 2020.06.11 ~ 2020.09.07 (약 3개월)

4. 나의 역할 : 딥러닝을 통한 이미지 데이터 분석

5. 팀 이름 : 누구 없소

6. 팀원 구성 : 컴퓨터 전공(개발), 경영학 전공(기획), 디자인 전공, 통계학 전공(분석), 산업수학 SW 전공(분석)

#
**이 코드들은 부산대학교 2020 창의 융합 해커톤 출품작 중 일부입니다.**
#
자신 주변의 카페, 음식점의 포화 여부를 고객에게 보여줌으로써 고객의 시간 절약, 그리고 소상공인의 판매 촉진을 위한 어플리케이션을 개발하는 "누구없소" 팀의 일원으로 활동하였었고,

딥러닝을 기반으로 한 CCTV영상 데이터를 분석하여 카페 및 음식점의 빈자리 개수를 알려주는 어플을 만들어 최우수상을 수상하였습니다. 

#
저는 팀 내에서 아이디어 기획 및 CCTV 화면을 캡처하여 딥러닝 모델을 통해 영상 속 사람이 몇 명인지 알아내는 부분을 맡았습니다.

#
영상 속 사람 수를 파악하기 위하여 keras 및 tensorflow를 이용하였고, VGG16 모델을 참조하여 자체 딥러닝 모델을 만들었습니다.

또한 imageai의 ObjectDetection 모델도 사용하였습니다.

자체 개발 모델과 ObjectDetection 모델의 결과를 종합하여 최종 결과를 return하는 알고리즘을 개발하였고,

최종 코드를 종합 후 pyInstaller를 통해 exe 파일을 생성하였습니다.


### 홍보영상 링크
https://www.youtube.com/watch?v=69w_5H0n9P8&feature=youtu.be
