## Tensorflow 사용 시 GPU 세팅 방법('17.11.18 시점 기준) ##

① CUDA를 설치해야 합니다. [CUDA](https://developer.nvidia.com/cuda-downloads)에서 본인 컴퓨터에 맞는 사양으로 설치하시면 됩니다.

② cuDNN 파일을 컴퓨터에 붙어넣어야 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;② - ⓐ [cuDNN](https://developer.nvidia.com/cudnn) 에서 컴퓨터 사양에 맞는 버전을 다운 받습니다.

&nbsp;&nbsp;&nbsp;&nbsp;② - ⓑ 다운 받은 파일의 압축을 풀어 cuda/bin, cuda/incldue, cuda/lib의 폴더와 파일을 다음 경로에 붙여 넣습니다. (아래 경로 참조 하시어 컴퓨터 실제 경로에 맞도록 붙여넣습니다.)

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0 (또는 9.0)

③ 제어판 내 시스템 => 고급 시스템 설정 => 시스템 속성 의 고급 탭 => 환경 변수 에서 아래 경로를 추가합니다. 경로 추가 시 어떤 path에서도 추가된 경로는 접근 가능해 집니다.

&nbsp;&nbsp;&nbsp;&nbsp;③ - ⓐ : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0\bin

&nbsp;&nbsp;&nbsp;&nbsp;③ - ⓑ : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0\libnvvp

④ cmd 에 ***pip install --upgrade tensorflow-gpu*** 를 입력하여 tensorflow gpu 버전을 설치합니다

⑤ Tensorflow 사용 시 GPU 사용하고 있는지 확인하는 방법은 GPU 사용량을 보면 되지만 GPU 사용해야만 run되는 코드를 run 해보면 쉽게 확인할 수 있습니다.

    import tensorflow as tf
    with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    
    with tf.Session() as sess:
    print (sess.run(c))

