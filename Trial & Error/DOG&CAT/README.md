## [1] 180103_ custom_ data_ cnn

- 사용 데이터 : 개/고양이 약 2,000장
- 사용 모델 : ResNet-14
- 특이 사항 : 없음
- 정확도 : 0.65 ~ 0.7 예상
- 평가 결과 : 데이터 수가 작아서 정확도가 매우 낮고 overfitting이 심함.

## [2] 180104_ custom_ data_ cnn

- 사용 데이터 : 개/고양이 약 45,000장
- 사용 모델 : ResNet-14
- 특이 사항 : **Image Augmentation 적용**하여 2,000장 이미지를 45,000장으로 증가시킴
- 정확도 : 0.8 ~ 0.85 예
- 평가 결과 : Image Augmentation을 이용하여 정확도↑, overfitting↓ 효과 확인

## [3] 180105_ custom_ data_ cnn

- 사용 데이터 : 개/고양이 dir 45,000장
- 사용 모델 : ResNet-14
- 특이 사항 : Cyclic Learning Rate 적용
- 정확도 : 0.8 ~ 0.85 예상
- 평균 train 정확도 : 86%, 평균 validation 정확도 : 78% (30 epoch 기준)
- 평가 결과 : Cyclic Learning Rate 적용 시 val.acc이 안정적으로 움직였다. L.R 최적화 중 가장 효과가 있어 보인다.
