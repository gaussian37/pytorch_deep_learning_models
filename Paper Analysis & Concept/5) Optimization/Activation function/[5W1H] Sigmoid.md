# ① WHEN & WHEN #

<span style="background-color: #FFFF00">Activation function</span>으로 각 layer 에서의 행렬곱, 합성곱등의 연산을 한 뒤 사용합니다.

# ② WHAT #

출처 : [http://taewan.kim/post/sigmoid_diff/](http://taewan.kim/post/sigmoid_diff/)

![1](https://i.imgur.com/iUhwTgy.jpg)

Sigmoid 함수는 S자와 유사한 완만한 시그모이드 커브 형태를 보이는 함수입니다.<br> 
Sigmoid는 대표적인 Logistic 함수입니다. Sigmoid 함수는 모든 실수 입력 값을 <span style="background-color: #FFFF00"> 0보다 크고 1보다 작은 미분 가능한 수로 변환</span>하는 특징을 갖습니다.<br>
모든 입력에 대하여 sigmoid는 S와 같은 형태로 미분 가능한 0 ~ 1 사이의 값을 반환하기에 Logistic Classification과 같은 분류 문제의 가설과 Cost Function에 많이 사용됩니다. sigmoid의 반환 값은 <span style="background-color: #FFFF00">확률형태(0 ~ 1사이 값)이기 때문에 결과를 확률로 해석</span>할 때 유용합니다. <br>
딥러닝에서는 노드에 임계값을 넘을 때만 출력하는 활성 함수로도 이용됩니다.

- Sigmoid 함수의 정의<br>

![2](https://i.imgur.com/9OfGKe0.png)

- Sigmoid 함수 미분 결과<br>

![3](https://i.imgur.com/VnZ04ke.png)

![4](https://i.imgur.com/uFPxNq2.png)

# ③ WHY #

Activation function 중의 하나로 비 선형성(Non-linearity)를 주기 위해 사용합니다. 하지만 sigmoid function을 Activation function으로 사용하는 것을 추천하지 않습니다. 사용 하지 않는 이유는 아래 3가지 이유를 확인 하시기 바랍니다.<br> 
그럼에도 불구하고 Sigmoid 함수는 <span style="background-color: #FFFF00">
미분 결과가 간결하고 사용하기 쉬우므로 초기에 많이 사용</span>되었습니다. 머신 러닝에서 Sigmoid함수는 가설과 학습에서 사용되었습니다. 학습에 사용될 때는 Sigmoid를 미분한 결과가 사용되고 <span style="background-color: #FFFF00">sigmoid는 미분 결과를 프로그래밍하기 쉽기에 인기가 더욱 높았습니다.</span>

<span style="color:red">**※ Sigmoid function의 문제점**</span>

![5](https://i.imgur.com/D4LRtn3.png)

① 입력 값이 일정 범위의 safety zone을 넘어가게 되면 0 또는 1로 수렴하게 되고 gradient(경사값) 또한 0으로 수렴해 버리게 됩니다. 양 끝이 평평해 지기 때문입니다.<br>
② sigmoid function의 범위가 [0, 1]입니다. 이 때문에 output의 중앙값이 0이 아니게 됩니다. 0을 기준으로 데이터가 분포하게 되었을 때가 이상적인데 Sigmoid에서는 하나의 단점이 됩니다.<br>
③ Relu와 비교해 보았을 때, exp() 연산에 많은 cost가 듭니다.<br>

좀 더 자세히 살펴보겠습니다.

![6](https://i.imgur.com/aeIEqpB.png)

파란색 부분이 safety zone 입니다. <span style="background-color: #FFFF00">safety zone을 넘어서게 되면 gradient 자체가 0으로 수렴</span>하게 됩니다.

![7](https://i.imgur.com/8PM9D1y.png)

Sigmoid가 <span style="background-color: #FFFF00">Non-zero centered 분포</span>를 가지게 되는 문제는 만약 input neuron이 항상 positive 라면 w에 대한 gradient가 항상 positive 또는 negative가 되어 학습이 잘 안될 수 있습니다. 학습하기 좋으려면 <span style="background-color: #FFFF00">cost function이 minimize 되는 지점을 다양한 gradient 크기로 찾아가야 하는데 항상 positive/negative 하면 minimize 지점을 찾기가 힘들어 집니다.</span>

![8](https://i.imgur.com/Ghzqy0y.png)

exponential 자체가 상당히 계산하기 비싼 term 이기 때문에 forward/backward 시 계산이 오래 걸립니다. (6배 정도 느린 것으로 알려져 있습니다.)

# ④ HOW #

- tensorflow 코드<br>
	`tf.sigmoid(x, name = None)`

- Keras 코드<br>
 activation option에 sigmoid 입력<br>
	`model.add(Dense(1, activation = 'sigmoid'))`