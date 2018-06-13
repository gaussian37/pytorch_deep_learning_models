# variable explorer in Jupyter #
![1](https://i.imgur.com/FIApm7Y.png)

1. 주피터 노트북이 켜져 있으면 서버를 닫아 줍니다.
2. 다음 두가지 방법 중 하나를 이용하여 설치합니다.<br>conda를 사용 한다면 :  `conda install -c conda-forge jupyter_contrib_nbextensions`</br> pip를 사용 한다면 :`pip install jupyter_contrib_nbextensions` <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`jupyter contrib nbextension install --user`
3. 설치된 extension을 Activation 시킵니다. : `jupyter nbextension enable varInspector/main`
4. 주피터 노트북을 실행합니다. : `jupyter notebook`
5. 다음 아이콘을 클릭하여 Inspector를 실행합니다.</br>
![2](https://i.imgur.com/u5Mp70m.png)



    