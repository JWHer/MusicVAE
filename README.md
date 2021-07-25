# Music VAE

- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music 기반으로

# 1일차
개발환경 세팅/ 논문 리뷰

## 개발환경 세팅

* 개발환경 선택 - WSL ubuntu18.04
빠르게 개인 노트북에 갖추어진 인프라 활용  
추후 모델 학습에 단점이 있음
<br/>

* library 설치
*magenta에 필요한 라이브러리 설치*  
<br/>

    installation [script](https://github.com/magenta/magenta/blob/main/magenta/tools/magenta-install.sh) 가 존재(conda) 하지만 [colab](https://colab.research.google.com/github/magenta/magenta-demos/blob/master/colab-notebooks/MusicVAE.ipynb#scrollTo=0x8YTRDwv8Gk) 예제에서 사용한 라이브러리 기반으로 직접 환경 구성

    ```shell
    # apt update, set superuser ...
    $ apt install libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev
    ```
<br/>

* python 세팅 - venv
<br/>

    ```shell
    # pip3 install virtualenv
    $ python3 -m venv $work_dir

    # activate/deactivate
    $ . $work_dir/bin/activate
    $ deactivate

    # required
    $ pip3 install tensorflow pyfluidsynth magenta
    ```
<br/>

* 논문

    [링크추가]()  
    전반적인 내용 파악(abstract) ~~프린트만 한듯~~
<br/>

## 2일차
데이터 준비/ 논문 리뷰  

[[유튜브]Music Generation with Magenta](https://www.youtube.com/watch?v=O4uBa0KMeNY)

* 데이터 다운로드  
    [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)
    <br/>

* 데이터셋 만들기 - tfrecord  
    magenta [스크립트](https://github.com/magenta/magenta/tree/main/magenta/scripts) 이용

    ```shell
    INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>

    # TFRecord file that will contain NoteSequence protocol buffers.
    SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

    convert_dir_to_note_sequences \
    --input_dir=$INPUT_DIRECTORY \
    --output_file=$SEQUENCES_TFRECORD \
    --recursive
    ```
    <br/>

* RNN 데이터셋 - 생략  
    [시퀀스](https://github.com/magenta/magenta/tree/main/magenta/models/drums_rnn#create-sequenceexamples)
    [모델학습]()
    <br/>

* VAE
    [학습](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae#training-your-own-musicvae)  

    학습을 위해 `configs.py` 파일을 수정/작성해 준다. 변경한 설정을 실행하려면 `import configs`로 `music_vae_train.py`를 수정하고 수정한 파일을 실행시켜야 한다. ~~너무 당연한건가~~

    ```shell
    # --num_steps 플래그
    music_vae_train --config=$config --run_dir=$run_dir --mode=train --examples_path=data/tfrecord/drummer1/session1.tfrecord
    ```
    <br/>

* 논문  
    hierarchical decoder의 nueral architecture를 파악해서 설정에 반영  

    ELBO 등 수학적 의미 학습
    <br/>

## 3일차
학습 구현 확인/ 논문 리뷰

* 추론
    ```shell
    # --run_dir 플래그 사용
    music_vae_generate \
    --config=cat-drums_2bar_small \
    --run_dir=$run_dir \
    --mode=sample \
    --num_outputs=3 \
    --output_dir=data/generated/
    ```

    WSL 팁  
    `explorer.exe .`: 탐색기 실행
    귀찮게 `scp`나 `ftp`, `sftp` 설정을 안해도 된다.

    <br/>

* 논문  
    학습 내용 전자electronic 문서화
    <br/>

## 4일차
학습 잘 안됨 확인/ 레포지토리 작업

* 학습 잘 안됨 확인  
    논문 내용 반영해서 config 수정, 재학습
    <br/>

* 레포지토리 작업  
    gitub push
    <br/>

## 5일차

* 추론  
    ```shell
    python3 music_vae_generate.py \
    --config=hier-drums_4bar \
    --run_dir=/mnt/q/model/v2/ \
    --mode=sample \
    --num_outputs=3 \
    --output_dir=/home/jwher/data/generated/
    ```
    <br/>

* 평가  
    ```shell
    python3 music_vae_train.py --config=hier-drums_4bar --run_dir=/mnt/q/model/v2/ --mode=eval --examples_path=/home/jwher/data/tfrecord/drummer1/session1.tfrecord
    ```

## 6일차
    (없음)

## 7일차
설명

### 이론
ELBO와 KL Divergence
(+추가)

### 데이터셋과 전처리  
* MIDI 파일 사용  
* 한 샘플은 16개 note  
1bar = 4bit =16note

cat-drums_2bar_small모델을 바탕으로 만듬  
```python
CONFIG_MAP['hier-drums_4bar_small'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
                level_lengths=[4, 16],
                disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=64,
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            free_bits=48,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=4,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)
```

### 학습  
* Encoder (Bidirection) 입력  
    [batch_size, max_sequence_length, input_depth]

    batch_size: 한번에 학습할 샘플 크기(512)

    max_sequence_length: 가능한 연속 길이(16x4)

    input_depth:  
    ```python
    #data.py
    if roll_input:
        input_depth = num_classes + 1 + add_end_token
    ```
    기본 드럼 종류: 9개 [kick, snare, closed_hh, open_hh, low_tom, mid_tom, hi_tom, crash, ride]  
    roll_input: True(1)  
    add_end_token: False(0)  
    input_depth: 9+1+0=10  
    따라서 16^10개의 악보가 가능하다.  

    MIDI 파일은 note당 input_depth는 4이다(?). 각 노트의 파라미터는  
    [ pitch, velocity, start_time, end_time ]이다.  
    *확실한 출처 필요*  

    [ 512x64x4 ] 차원으로 시간 순서에 따라 LSTM셀 1개는 [ 1x512x4 ] 입력을 받음.  
    인코더 RNN은 양방향 LSTM으로써 64개의 LSTM셀(64bar 음악)이다. 이는 [ 1x512x4 ]으로 양방향 연결되어 있다.  
    
    > mu = W_hmu * h_T + b_mu  
    > sigma = log(exp(W_hsigma + b_sigma) + 1)  
    *W: 가중치 행렬, b: 편향 벡터*
    의 정규분포(Gaussian distribution)로 표현될 수 있음  
    
    각각 512크기인 LSTM 레이어 한개를 가집니다.
    ```python
    # lstm_models.py - BidirectionalLstmEncoder
    lstm_utils.build_bidirectional_lstm(
        layer_sizes=hparams.enc_rnn_size,
        dropout_keep_prob=hparams.dropout_keep_prob,
        residual=hparams.residual_encoder,
        is_training=is_training)
    # lstm_utils.py
    def build_bidirectional_lstm(
        layer_sizes, dropout_keep_prob, residual, is_training):
    ```
    <br/>

* Decoder (Hierarchical)  
    긴 시퀀스를 학습하기 위해 conducter 레이어를 추가로 사용.
    level_length = [ 4, 16 ]  
    첫번째 레벨에 4개의 LSTM블록이 있고 두번째 레벨에선 각 첫번째 블록마다 16개의 LSTM블록이 있음. (4*16 = 64)  
    각 LSTM블록의 셀 크기는 dec_rnn_size인 두개의 256셀 이다.  

* free_bits, max_beta
    KL loss 값을 줄이기 위해서 free_bits를 늘리거나 max_beta를 줄일 수 있다.  
    하지만, 잠재적으로 나쁜 random sample을 얻을 수 있다.

### 생성  

1. z는 dense layer로 전달받는다. Conductor는 512LSTM 단위, state size는 256, 256이므로
flatten size는 [512, 256, 256]이다.  
1. 출력 dense layer는 flatten 상태의 합 512*1024이다.
1. 분할?
1. 각 분할이 lstm state size로 초기 상태
1. 초기 상태에서 depth wize 디코딩이 되어 아래 레이어로 전달된다.
1. Conductor의 출력은 다음 레벨의 decoder 입력이 된다.
1. decoder는 단방향 lstm을 수행한다.
1. Conductor는 core decoder와 독립적으로 latent vector가 긴 길이 생성에 도움을 준다.
1. reconstruction loss는 cross entropy를 사용해 최종 출력과 실제 값을 비교한다.
1. Adam optimizer를 학습에 사용한다.

### 결과
[Generated](https://github.com/JWHer/MusicVAE/tree/main/data/generated)

<!-- 
references  
[medium](https://medium.com/@musicvaeubcse/musicvae-understanding-of-the-googles-work-for-interpolating-two-music-sequences-621dcbfa307c)  
[VAE](https://deepinsight.tistory.com/127)
-->