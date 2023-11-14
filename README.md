 # 항만내의 TTP구간 불법침입 및 안전사고 종합대응시스템   
    
 - DSBA 예비프로젝트
 1. 프로젝트 선정 배경
 2. 시스템 흐름
 3. 시스템 설계
 4. 마무리

 ## 프로젝트 선정 배경
<img src="https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/0dd1c4dd-9c88-443f-99fc-1e82f95270d9" width=700 height = 400/>   

- 해양경찰청에서 제공하는 통계에 따르면 해양사고 발생지 중 'TTP 구간'은 해안가와 항포구에 이어 상위권에 속한다.
- 오른쪽 그래프는 매년 사고 발생 수의 통계자료이다. TTP 구간 내에서 사고 발생 수 역시 지속적으로 발생하고 있다는 것을 알 수 있다.
   
<img src = "https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/f77d3091-cfef-4285-b310-48a8605107e2" width = 700 height = 400/>   

- 사망사고 역시 매년 지속적으로 발생하고 있고, 해안가를 다음으로 평균 사망자가 많은 곳도 'TTP 구간'이다.
   
■ 이를 방지하기 위해 국가적인 차원에서 관련 법(항만법 제28조 제2항 금지행위등)을 방안으로 제시하기도 하였고, 항만 관련 기관에서는 순찰 업무를 수행하는 부서를 편성하여 운영하고 있으나, 여전히 불법침입 및 사고 사례들이 발생하고 있는 실정이다.
   
■ 이러한 사고들을 현행 대응방안보다 더욱 효과적으로 예방할 수 있는 방법에 대하여 생각해보았고, 3가지 핵심 기능으로 구성된 종합 대응 시스템을 기획하여 제안한다.
   
 ## 시스템 흐름
 <img src = "https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/0726fb2a-dc8b-48bd-992a-b6a3739fe4ae" width = 700 height=600/>

1) 지능형 CCTV
2) 침입시간 예측 알고리즘
3) 장소 추천 챗봇   
- 출입제한구역의 지능형 CCTV에 침입자가 탑지되면 경고방송이 송출, 관련 정보를 데이터프레임 형태로 저장
- 감시기록 데이터(시계열 데이터) 전처리 및 머신러닝 모델 학습과정을 통해 침입 다발 시간대를 예측
- 이를 통하여 효율적인 순찰스케줄을 계획할 수 있도록 도움
- 출입제한구역에 출입한 침입자에게 출입이 가능한 TTP구간이나 낚시터 정보를 제공하여 재침입을 방지

## 시스템 설계
### 1. 지능형CCTV
![지능형cctv](https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/4409956b-0c8a-4a47-b167-e10527b1e6fc)

### 2. 시간예측 알고리즘
2-1 Arima
![arima](https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/19c46216-f20a-4451-a6d1-d6cba037709f)

2-2 LSTM
![lstm2](https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/492ffe84-a3fe-40c2-b8a2-9ce0280f9c83)

2-3 GRU
![gru2](https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/1afa8177-1f9d-48e8-aa31-fe1df1fac6a0)

      <종합>
||MSE|RMSE|
|:---:|:---:|:---:|
|ARIMA|3.47409|1.8638|
|**LSTM**|0.41007|0.6403|
|GRU|0.43417|0.6589|

   
### 3. 장소 추천 챗봇
   
3-1 형태소 분석기(KoNLPy-Kkma)
![형태소분석1](https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/2b92986c-ae5d-4ae5-b6f3-5c4e2420cb34)


3-2 결과(django기반 웹 프로토타입)
![형태소분석2](https://github.com/jong4499/DSBA_pre_PJT/assets/141287150/1e478862-e69b-43e6-9062-9e7a1f942516)

