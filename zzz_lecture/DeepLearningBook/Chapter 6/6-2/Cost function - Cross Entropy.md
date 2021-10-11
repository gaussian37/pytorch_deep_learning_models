cost function에 관하여 공부하면서 궁금증이 들었습니다.

왜 cost function으로 cross entropy를 사용할까요?

많은 책들이 MSE(Mean Squared Error)를 이용하여 정답값과 예측값의 차이를 설명하면서 정작 MSE를 사용하지 않고 cross entropy를 사용할까 라는 생각이 들어 정리를 해보았습니다.

@Terry TaeWoong Um 님 자료 참조

----------
# ① Entropy란 무엇일까? #

[동영상 링크](http://serviceapi.nmv.naver.com/flash/convertIframeTag.nhn?vid=D774A5D242BF7052317B409E628E7D1748D4&outKey=V122d0b64ace7741db1a48478d873679f954910f99dfbdea27b188478d873679f9549&width=544&height=306)

Entropy란 정보를 최적으로 인코딩하기 위해 필요한 bit의 수 입니다. 따라서 bit 수가 많아지면 그 만큼 정보량이 많아지는 것입니다. 이 때 bit 수는 최적으로 압축 하였을 때의 bit수를 가지고 entropy를 표현합니다. 그러면 어떻게 해야 정보를 최적으로 압축을 시킬 지 알아보겠습니다.
 
예를 들어 요일을 표현할 때에는 최대 7일이므로 2<sup>3</sup> = 8, 즉, 최소 3bit 가 필요합니다. 따라서 요일을 표현할 때의 정보량은 3 bit입니다. 간단하게 log<sub>2</sub>N으로 표현할 수 있습니다.

하지만 요일의 발생확률은 동일합니다. 만약 각각의 발생 확률이 다르다면 어떻게 계산할까요?
만약 40개의 문자(A, B, C, D, ..., 11, 12, 13, 14)를 bit로 전송한다면 log<sub>2</sub>40 = 5.3이 되어 6bit로 표현할 수 있습니다. 

그런데 만약에 발생 확률이 다른것으로 가정하여 볼 때, 위의 문자를 병으로 보면 A, B, C, D가 전체의 22.5% 씩 전체 90% 확률로 발생한다고 가정해 봅시다. 예를 들어, 병원에 찾아오는 사람들의 90%가 정상입니다. 그리고 10%만 희귀병으로 진단이 내려 집니다. 90%의 확률에 해당하는 병에는 bit를 조금만 발생 시킵니다. 왜냐하면 빈번하게 발생하므로 빈도(확률)가 높은 병에 대한 cost를 줄여야 합니다. 반대로 10%의 확률에 해당하는 병에는 cost를 비싸게 처리해 줍니다.
따라서 A, B, C, D에 대한 bit 수를 줄여버리고 나머지 36개에는 bit를 많이 둡니다.

① 1st bit : A, B, C, D 인지 아닌지 확인


- Yes : 추가로 2bit 더 필요(A,B,C,D 구분)


- No : 추가로 6 bit 더 필요 (log<sub>2</sub>36 = 5.xx)

따라서 필요한 비트수 = 0.9 x (1+2) + 0.1 x (1+6) = 3.4 (bit)가 필요합니다.

즉, 발생 확률이 달라지면 정보량이 달라지고(발생 확률 같을 때 : 5.3 bit, 다를 때 : 3.4 bit), Entropy는 각 label들의 확률 분포함수와 같이 표현이 됩니다. 발생 확률이 크면 cost를 줄이고 확률이 작으면 cost를 늘이게 됩니다.

![1](https://i.imgur.com/4Wzooyn.png)

위 수식에서는 y<sub>i</sub> 는 예제의 0.9(A, B, C, D 각각 0.225)와 0.1에 해당합니다. 

따라서 H(x) = -( 4 x 0.225 x log<sub>2</sub>(0.225) + 36 x 0.0028 x log<sub>2</sub>(0.0028)) = 2.72 bit가 됩니다. 조금 전 3.3bit보다 더 줄어들게 되었고 위 식처럼 표현하는 것이 최적입니다.

### 정리하면, 내가 가진 정보의 양을 최적으로 압축시켰을 때 나오는 bit 수가 Entropy 입니다. Entropy의 식은 확률 분포를 구하는 것과 같이 확률(y<sub>i</sub>) x 


# ② cross-entropy와 kl-divergence #

[동영상 링크](http://serviceapi.nmv.naver.com/flash/convertIframeTag.nhn?vid=6C038F5E1100D437440FD1D7AE3E5348974E&outKey=V1235408ebc92af5470988010450a1a9a7a5d7dc9d38e1b1279e68010450a1a9a7a5d&width=544&height=306)

다시 한번 더 Entropy를 어떻게 표현하는지 알아보겠습니다.

![1](https://i.imgur.com/4Wzooyn.png)

- N개의 사건을 분류하기 위하여 log<sub>2</sub>N bit가 필요합니다.
- 빈번한 케이스(발생 확률 大)는 bit 수를 줄이고 드문드문한 케이스는 bit수를 늘린다.

먼저 시그마 내부 term 중 ![4](https://i.imgur.com/GJPdrIq.png)에 대하여 알아보겠습니다.
y<sub>i</sub> 는 발생 확률이고, ![4](https://i.imgur.com/GJPdrIq.png)는 information gain 즉, 정보 획득량 입니다.

예를 들어 용의자를 찾았을 때, 용의자의 성을 알게되었다면, 김씨와 엄씨 중 어떤 성씨가 정보량을 많이 가지게 될까요?

정답은 엄씨 입니다. <span style="color:red">즉, 정보의 희귀성과 정보양은 비례합니다. (정보의 희귀성과 발생 확률에는 반비례)</span>

### Cross Entrypy ###
![](https://i.imgur.com/vIsys7R.png)

이제 Cross Entropy에 대하여 알아보도록 하겠습니다. 먼저 Entropy는 최적의 정보량을 나타내는 개념이었습니다.

만약에 최적이 아니라 <span style="color:red">잘못된 정보를 사용하여 information gain을 얻었다고</span> 가정해 봅시다. 이렇게 얻어진 최적의 bit 값은 실제 최적의 bit 값 보다 더 큰 값을 가지게 됩니다. (Entropy의 목적은 정보를 표현할 때 필요한 bit를 최대한 압축하는 것이므로 오 정보로 압출을 잘못 하면 최적일 때 보다 bit 값이 커지게 됩니다.)
위의 수식을 연계하면 q(x) 라는 잘못된 확률 정보를 통해서 얻은 Entropy 값에 해당합니다. 
