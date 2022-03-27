python3 inference.py -c  \
configs/default.yaml -p checkpoints/test/test_fastspeech_d0e9a25_570k_steps.pyt \
--text "A skunk sat on a stump and thunk the stump stunk, but the stump thunk the skunk stunk" --ref_mel ./ref_wav/Eleanor.npy --out wav