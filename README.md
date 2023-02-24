# AdjustSpeechPower
Equalize the power in each voice interval.

## Algorithm
音声区間を検出し，各区間の振幅の二乗平均平方根 (Root Mean Square: RMS) の値を統一する．
* 音声区間の検出には，事前学習済み音声区間検出器[SILERO VOICE ACTIVITY DETECTOR](https://pytorch.org/hub/snakers4_silero-vad_vad/)を使用した．
* 音声区間のRMSは0.05 (default) となるように，各区間で振幅を定数倍した．
* 非音声区間のRMSは，音声区間のRMS (defaultで0.05) に対してSignal-to-Noise Ratio (SNR) が30dB (default) となるように振幅を定数倍した．
* 音声区間検出器が誤った判定をする場合を観測したため，1つのファイルで検出された各音声区間について，
その区間のRMSがファイル全体で検出された音声区間のRMSの平均の20% (default) 未満である場合は，その区間を非音声区間とみなした．
* 処理対象の信号は全て，サンプリング周波数が48kHz，量子化ビット数が16bit，複数チャネルの信号は1ch目のみを使うことでモノラル信号に，事前に変換した．
