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