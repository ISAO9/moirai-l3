# Numbers quoted in the supplementary prose, against the logs

| Quantity | In the text | In the log | |
|---|---|---|---|
| S1 両極端は forge_19 と clearfield_mw6 | forge_19 / clearfield_mw6 | forge_19 / clearfield_mw6 | OK |
| S1 flagged は 0〜21 % | 0-21 % | 0.5-20.9 % | OK |
| S1 forge_19 flagged = 1 | 1 | 1 | OK |
| S1 forge_19 earthquakes = 213 | 213 | 213 | OK |
| S1 clearfield_mw6 flagged = 128 | 128 | 128 | OK |
| S1 clearfield_mw6 earthquakes = 612 | 612 | 612 | OK |
| S1 反転検査を切るのは aneth だけ | aneth | aneth | OK |
| S1 窓端 60 サンプル以内（flagged）= 8 % | 8 | 8.0 | OK |
| S1 窓端 60 サンプル以内（健全）= 1 % | 1 | 1.0 | OK |
| S1 生の P moveout の最大 = 909 ms | 909 | 909  [pnr-1 365, mseel_3h 767, mseel_5h 692, clearfield_mw6 909, pnr-2 608, clearfield_mw4 648, aneth 554, forge_19 82] | OK |
| S1 350 ms を超えるのは 8 サイト中 7 | 7 | 7 | OK |
| S1 forge_19 の除外は 2 件 | 2 | 2 | OK |
| S1 forge_19 array 0.516 -> 0.517 | 0.517 | 0.5171412830736221 | OK |
| S1 forge_19 per-trace 0.856 -> 0.858 | 0.858 | 0.8580082502694326 | OK |
| S1 8 サイトすべて判定不変 | 8/8 | 8/8 | OK |
| S1 清浄化は全サイトで上がる | all > 0 | 最小 +0.001 (forge_19 array) | OK |
| S1 上昇の最大 = 0.074 | 0.074 | 0.0744 | OK |
| S1 その最大は clearfield_mw6 | clearfield_mw6 | clearfield_mw6 (pertrace) | OK |
| S1 再現検査は 16 件 | 16 | 16 | OK |
| S2 動きの最大 = 0.013 | 0.013 | 0.0127 | OK |
| S2 同一 6 件 / tie→array 2 件 | 6 / 2 | 6 / 2 ['mseel_5h', 'clearfield_mw4'] | OK |
| S2 方向転換ゼロ | 0 | 0 | OK |
| S2 forge_19 差 = -0.332 | -0.332 | -0.3321624046947884 | OK |
| S2 forge_19 CI 下限 = -0.355 | -0.355 | -0.3549796150567523 | OK |
| S2 forge_19 CI 上限 = -0.310 | -0.31 | -0.3098244888301939 | OK |
| S2 aneth 差 = -0.055 | -0.055 | -0.05536549821771075 | OK |
| S2 aneth CI 下限 = -0.065 | -0.065 | -0.06451488173166257 | OK |
| S2 aneth CI 上限 = -0.047 | -0.047 | -0.04652947250709768 | OK |
| S2 pnr-2 差 = +0.036 | 0.036 | 0.03568847705701911 | OK |
| S2 pnr-2 CI 下限 = +0.027 | 0.027 | 0.026632906698454424 | OK |
| S2 pnr-2 CI 上限 = +0.046 | 0.046 | 0.045639597224064715 | OK |
| S2 forge_19 paired array = 0.524 | 0.524 | 0.5237220186498376 | OK |
| S3 解析対象 = 44 ラン | 44 | 44 | OK |
| S3 学習可能パラメータ = 7,991,491 | 7991491 | 7991491 | OK |
| S3 dev split で停止 = 41 ラン | 41 | 41 | OK |
| S3 停止しなかった3ランの最終改善は 4, 9, 18 エポック前 | 4, 9, 18 | 4, 9, 18 | OK |
| S3 ベンチマークの記録数 = 9,803 | 9803 | 9803 | OK |
| S3 1ランの訓練記録数 下限 = 4,769 | 4769 | 4769 | OK |
| S3 1ランの訓練記録数 上限 = 5,932 | 5932 | 5932 | OK |
| S3 1ランの訓練トレース数 下限 = 58,464 | 58464 | 58464 | OK |
| S3 1ランの訓練トレース数 上限 = 72,420 | 72420 | 72420 | OK |
| S3 trend を持つ = 43 ラン | 43 | 43 | OK |
| S3 trend が負 = 30 ラン | 30 | 30 | OK |
| S3 trend の中央値 = -0.0018 | -0.0018 | -0.0017629897228201852 | OK |
| S3 trend / 変動 の中央値 = -0.36 | -0.36 | -0.3616255196828102 | OK |
| S3 epoch 間変動の中央値 = 0.0045 | 0.0045 | 0.004529769877537379 | OK |
| S3 dev スコアの中央値 = 0.93 | 0.93 | 0.9319021350986881 | OK |
| S3 最大からの落差の中央値 = 0.011 | 0.011 | 0.011108666849518667 | OK |
| S3 損失比が使える = 42 ラン | 42 | 42 | OK |
| S3 損失比の下限 = 22 | 22 | 21.617210066588907 | OK |
| S3 損失比の上限 = 41 | 41 | 41.32377791671557 | OK |
| S3 損失比の中央値 = 29 | 29 | 29.452089893113232 | OK |
| S3 記録の総数 = 9,803 | 9803 | 9803 | OK |
| S3 地震の総数 = 3,421 | 3421 | 3421 | OK |
| S3 雑音窓の総数 = 6,382 | 6382 | 6382 | OK |
| S3 1ランの訓練地震数 下限 = 1,527 | 1527 | 1527 | OK |
| S3 1ランの訓練地震数 上限 = 2,002 | 2002 | 2002 | OK |
| S3 1ランの訓練記録数 下限 = 4,769 | 4769 | 4769 | OK |
| S3 1ランの訓練記録数 上限 = 5,932 | 5932 | 5932 | OK |
| S5 forge_19 array P: picks = 2,023 | 2023 | 2023 | OK |
| S5 forge_19 array P: on-phase = 0.181 | 0.181 | 0.1814137419673752 | OK |
| S5 forge_19 array P: cross-phase = 0.229 | 0.229 | 0.22886801779535343 | OK |
| S5 forge_19 array P: between = 0.482 | 0.482 | 0.48217317487266553 | OK |
| S5 forge_19 array P: median offset = +146 ms | 146.0 | 146.0 | OK |
| S5 forge_19 per-trace P: on-phase = 0.824 | 0.824 | 0.8243459690336359 | OK |
| S5 forge_19 per-trace P: cross-phase = 0.005 | 0.005 | 0.005339028296849973 | OK |
| S5 mseel_5h array P: on-phase = 0.913 | 0.913 | 0.9128486055776892 | OK |
| S5 forge_19 array S: on-phase = 0.824 | 0.824 | 0.8235294117647058 | OK |
| S5 forge_19 array S: cross-phase = 0.053 | 0.053 | 0.05310457516339869 | OK |
| S6 aneth: share below 10 ms = 86.6 % | 86.6 | 86.6 | OK |
| S6 aneth: events below 10 ms = 258 | 258 | 258 | OK |
| S6 aneth: events = 298 | 298 | 298 | OK |
| S6 other sites: maximum share = 2.4 % | 2.4 | 2.4 | OK |
| S6 four sites carry no event below 10 ms | 4 | 4 | OK |
| S4 forge_19 within-array array = 0.950 | 0.95 | 0.95038700062497 | OK |
| S4 forge_19 within-array per-trace = 0.952 | 0.952 | 0.952440098661029 | OK |
| S4 forge_19 LOSO array = 0.516 | 0.516 | 0.5157911478859388 | OK |
| S4 test records in all = 2,530 | 2530 | 2530 | OK |
| S4 最小の test split = 47 | 47 | 47 | OK |
| S4 最大の test split = 1,258 | 1258 | 1258 | OK |
| S4 forge_19 の差 LOSO = 0.340 | 0.34 | 0.3397426680209543 | OK |
| S4 forge_19 の差 within = 0.002 | 0.002 | 0.00205309803605902 | OK |
| S4 aneth の per-trace 優位 = 0.057 | 0.057 | 0.05709810840951812 | OK |
| S4 aneth の within 差 = +0.003 | 0.003 | 0.002853451375219085 | OK |
| S4 within の判定 array 2 / per-trace 1 / tie 5 | 2 / 1 / 5 | 2 / 1 / 5 | OK |
| S4 LOSO の判定 array 1 / per-trace 2 / tie 5 | 1 / 2 / 5 | 1 / 2 / 5 | OK |
| S4 within の差の最大 = 0.017 | 0.017 | 0.017364027450121944 | OK |
| S4 『forge_19 の差の20分の1』 | 20 | 19.6 | OK |
| S4 mseel_3h array の低下 = 0.040 | 0.04 | 0.040102430338616823 | OK |
| S4 mseel_3h per-trace の低下 = 0.042 | 0.042 | 0.041879338316284764 | OK |
