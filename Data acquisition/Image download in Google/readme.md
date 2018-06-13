# Google Image 크롤링 방법 #

① 웹 브라우저 개발자 모드로 들어갑니다. (F12) </br>
② 개발자 모드의 console 로 들어갑니다. </br>
③ github 내 get_url.js의 코드를 console 창에 입력하면 각 이미지들의 url 목록을 받을 수 있습니다. </br>

![1](https://i.imgur.com/aZLlZqw.png)

④ download_image.py를 다운 받아 urls.txt 파일과 동일 디렉토리에 배치합니다. </br>
⑤   `import request` </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`import cv2` </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;두가지 library를 import 하기 위하여 library를 설치합니다. </br></br>
※ 아래 설치 방법은 <span style="color:red">anaconda3를 설치하였을 때 기준</span> 입니다.

    conda install -c conda-forge opencv 
    pip install requests

⑥ 다운받은 python 파일을 다음과 같이 실행 합니다.</br>

    python download_images.py --urls urls.txt --output DIRECTORY

&nbsp;&nbsp;DIRECTORY에 파일을 실제 저장할 경로를 추가하면 됩니다.</br>
&nbsp;&nbsp;ex) python download_images.py --urls <span style="color:red">urls.txt</span> --output <span style="color:red">image/</span></br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ python 파일과 urls.txt가 동일 디렉토리에 있고 하부 디렉토리로 image 폴더가 있을 때, image 폴더내에 image 저장