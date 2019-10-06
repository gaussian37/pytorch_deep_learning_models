# Dense Layer #

출처 : 블록과 함께하는 파이썬 케라스 딥러닝

Dense 레이어는 <span style="background-color: #FFFF00">
입력과 출력을 모두 연결해 줍니다.</span><br>
예를 들어 입력 뉴런이 4개, 출력 뉴런이 8개 있다면, 총 연결선은 32개(4 x 8 = 32) 입니다.
각 연결선은 가중치(weight)를 포함하고 있는데, 이 가중치가 연결강도를 의미합니다. 현재 연결선이 32개이므로 가중치도 32개 입니다.

> ## 가중치가 높을수록 해당 입력 뉴런이 출력 뉴런에 미치는 영향이 크고, 낮을수록 미치는 영향이 작습니다. ##

예를 들어, 성별을 판단하는 문제에서 출력 뉴런의 값이 성별을 의미하고, 입력 뉴런에 머리카락 길이, 키, 혈액형 등이 있다고 가정하면, 머리카락 길이의 가중치가 가장 높고, 키의 가중치가 중간이고, 혈액형의 가중치가 가장 낮을 것입니다. <br>
<span style="background-color: #FFFF00"> 딥러닝 학습 과정에서 이러한 가중치들이 조정</span>됩니다.

이렇게 입력 뉴런과 출력 뉴런을 모두 연결한다고 해서 전 결합층이라고 불리고, Keras에서는 Dense라는 클래스로 구현이 되어 있습니다.

    Dense(8, input_dim = 4, activation = 'relu')

- 첫 번째 인자 : 출력 뉴런의 수를 설정합니다.
- input_dim : 입력 뉴런의 수를 설정합니다.
- activation : 활성화 함수를 설정합니다.
  - 'linear' : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
  - 'relu' : rectifier 함수, <span style="background-color: #FFFF00">은닉층</span>에 주로 쓰입니다.
  - 'sigmoid' : 시그모이드 함수, <span style="background-color: #FFFF00">이진 분류 문제에서 출력층</span>에 주로 쓰입니다.
  - 'softmax' : 소프트맥스 함수, <span style="background-color: #FFFF00">다중 클래스분류 문제에서 출력층</span>에 주로 쓰입니다.

Dense layer는 입력 뉴런 수에 상관 없이 **출력 뉴런 수를 자유롭게 설정**할 수 있기 때문에 **출력층으로 많이 사용**합니다.

예를 들어,<br> 1) **이진 분류** 문제에서는 **0과 1을 나타내는 출력 뉴런 하나**만 있으면 되기 때문에 아래 코드처럼 출력 뉴런이 1개이고, 입력 뉴런과 가중치를 계산한 값을 0에서 1 사이로 표현할 수 있는 활성화 함수인 'sigmoid'를 사용합니다.

	Dense(1, input_dim = 4, activation = 'sigmoid')

2) **다중 클래스분류 문제**에서는 **클래스 수만큼 출력 뉴런이 필요**합니다. 만약 세 가지 종류로 분류한다면, 아래 코드처럼 출력 뉴런이 3개이고 입력 뉴런과 가중치를 계산한 값을 각 클래스의 확률 개념으로 표현할 수 있는 활성화 함수인 **softmax**를 사용합니다.

	Dense(3, input_dim = 4, activation = 'softmax')

&nbsp;&nbsp;&nbsp;&nbsp;위 코드에서는 입력 신호가 4개이고, 출력 신호가 3개 이므로 시냅스 강도의 갯수는 12개 입니다.

3) Dense 레이어는 보통 출력층 이전의 은닉층으로 많이 쓰이고 영상이 아닌 수치자료 입력 시에는 입력층으로도 많이 쓰입니다. 이 떄 활성화 함수 'relu'가 주로 사용됩니다. 'relu'는 학습과정에서 back-propagation 시에 좋은 성능이 나오는 것으로 알려져 있습니다.

	Dense(4, input_dim = 6, activation = 'relu')

또한 입력층이 아닐 때에는 이전층의 출력 뉴런 수를 알 수 있기 때문에 input_dim을 지정하지 않아도 됩니다.<br>
아래 예를 보면 입력층에만 input_dim을 정의하였고 이후 층에는 input_dim을 지정하지 않았습니다.

	model.add(Dense(8, input_dim = 4, activation = 'relu'))
	model.add(Dense(6, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))

4개의 입력 값을 받아 이진 분류하는 문제는 아래와 같습니다.

	from keras.models import Sequential
	from keras.layers import Dense

	model = Sequential()

	model.add(Dense(8, input_dim = 4, activation = 'relu'))
	model.add(Dense(6, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))

케라스의 시각화 기능을 이용하여 구성된 레이어를 벡터 이미지형태로 볼 수 있습니다.

	from IPython.display import SVG
	from keras.utils.vis_utils import model_to_dot

	%matplotlib inline

	SVG(model_to_dot(model, show_shapes = True).create(prog='dot', format = 'svg'))

![1](https://i.imgur.com/t7YMdKm.png)




