**\[ 1. 데이터 수집 ]**

pip install autocar3g 로 해당 파일 source 다운로드

local에서 임의의 폴더 생성 후 

product 파일에서 해당 IP 주소에 맞는 주소설정.

keydrive.py 와 image.py 파일로 차량 데이터 수집



**\[ 2. 데이터 파싱 ]**

track\_dataset.py로 영상 데이터 파싱.

파싱된 사진 파일은 annotator 폴더의 annotator.py 파일에서 일일이 라벨링



**\[3. 모델 학습 ]**

model\_train.py로 라벨링된 데이터 학습 후 Track\_Model.h5 파일 생성



**\[ 4. 차량 주행 ]**

autodrive.py 코드에서 Track\_Model.h5 파일로 주행.



