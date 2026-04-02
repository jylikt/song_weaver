root@ubuntu:~# tree ~/xcodec
/root/xcodec
├── README.md
├── RepCodec
│   ├── LICENSE
│   ├── README.md
│   ├── examples
│   │   ├── data2vec_audio.py
│   │   ├── data2vec_feature_reader.py
│   │   ├── dump_feature.py
│   │   ├── feature_utils.py
│   │   ├── hubert_feature_reader.py
│   │   ├── tokens
│   │   │   ├── data2vec_base_l6_dev-clean.tokens
│   │   │   ├── data2vec_large_l18_dev-clean.tokens
│   │   │   ├── hubert_base_l9_dev-clean.tokens
│   │   │   ├── hubert_large_l18_dev-clean.tokens
│   │   │   ├── whisper_large_l32_dev-clean.tokens
│   │   │   └── whisper_medium_l24_dev-clean.tokens
│   │   ├── whisper_feature_reader.py
│   │   └── whisper_model.py
│   ├── repcodec
│   │   ├── RepCodec.py
│   │   ├── configs
│   │   │   ├── repcodec_dim1024.yaml
│   │   │   ├── repcodec_dim1280.yaml
│   │   │   └── repcodec_dim768.yaml
│   │   ├── layers
│   │   │   ├── conv_layer.py
│   │   │   └── vq_module.py
│   │   ├── modules
│   │   │   ├── decoder.py
│   │   │   ├── encoder.py
│   │   │   ├── projector.py
│   │   │   ├── quantizer.py
│   │   │   └── residual_unit.py
│   │   └── tokenize.py
│   ├── setup.py
│   ├── train.py
│   ├── train_configs
│   │   └── ex_dim768_mse.yaml
│   └── trainer
│       └── autoencoder.py
├── __pycache__
│   ├── post_process_audio.cpython-310.pyc
│   └── vocoder.cpython-310.pyc
├── decoders
│   ├── config.yaml
│   ├── decoder_131000.pth
│   └── decoder_151000.pth
├── descriptaudiocodec
│   └── dac
│       ├── __init__.py
│       ├── __main__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   ├── __init__.cpython-38.pyc
│       │   └── __init__.cpython-39.pyc
│       ├── compare
│       │   ├── __init__.py
│       │   └── encodec.py
│       ├── model
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-310.pyc
│       │   │   ├── __init__.cpython-39.pyc
│       │   │   ├── base.cpython-310.pyc
│       │   │   ├── base.cpython-39.pyc
│       │   │   ├── dac.cpython-310.pyc
│       │   │   ├── dac.cpython-39.pyc
│       │   │   ├── discriminator.cpython-310.pyc
│       │   │   └── discriminator.cpython-39.pyc
│       │   ├── base.py
│       │   ├── dac.py
│       │   └── discriminator.py
│       ├── nn
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-310.pyc
│       │   │   ├── __init__.cpython-39.pyc
│       │   │   ├── layers.cpython-310.pyc
│       │   │   ├── layers.cpython-39.pyc
│       │   │   ├── loss.cpython-310.pyc
│       │   │   ├── loss.cpython-39.pyc
│       │   │   ├── quantize.cpython-310.pyc
│       │   │   └── quantize.cpython-39.pyc
│       │   ├── layers.py
│       │   ├── loss.py
│       │   └── quantize.py
│       └── utils
│           ├── __init__.py
│           ├── __pycache__
│           │   ├── __init__.cpython-310.pyc
│           │   └── __init__.cpython-39.pyc
│           ├── decode.py
│           └── encode.py
├── final_ckpt
│   ├── ckpt_00360000.pth
│   └── config.yaml
├── models
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-39.pyc
│   │   ├── soundstream_hubert_new.cpython-310.pyc
│   │   └── soundstream_hubert_new.cpython-39.pyc
│   └── soundstream_hubert_new.py
├── modules
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-39.pyc
│   │   ├── conv.cpython-310.pyc
│   │   ├── conv.cpython-39.pyc
│   │   ├── lstm.cpython-310.pyc
│   │   ├── lstm.cpython-39.pyc
│   │   ├── norm.cpython-310.pyc
│   │   ├── norm.cpython-39.pyc
│   │   ├── seanet.cpython-310.pyc
│   │   ├── seanet.cpython-39.pyc
│   │   ├── transformer.cpython-310.pyc
│   │   └── transformer.cpython-39.pyc
│   ├── conv.py
│   ├── loss.py
│   ├── lstm.py
│   ├── norm.py
│   ├── seanet.py
│   └── transformer.py
├── post_process_audio.py
├── quantization
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-39.pyc
│   │   ├── core_vq_lsx_version.cpython-310.pyc
│   │   ├── core_vq_lsx_version.cpython-39.pyc
│   │   ├── distrib.cpython-310.pyc
│   │   ├── distrib.cpython-39.pyc
│   │   ├── vq.cpython-310.pyc
│   │   └── vq.cpython-39.pyc
│   ├── core_vq.py
│   ├── core_vq_lsx_version.py
│   ├── distrib.py
│   └── vq.py
├── semantic_ckpts
│   └── hf_1_325000
│       ├── config.json
│       ├── preprocessor_config.json
│       └── pytorch_model.bin
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-39.pyc
│   │   ├── ddp_utils.cpython-310.pyc
│   │   ├── ddp_utils.cpython-39.pyc
│   │   ├── utils.cpython-310.pyc
│   │   └── utils.cpython-39.pyc
│   ├── ddp_utils.py
│   └── utils.py
├── vocoder.py
└── vocos
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   ├── __init__.cpython-311.pyc
    │   ├── __init__.cpython-39.pyc
    │   ├── dataset.cpython-39.pyc
    │   ├── discriminators.cpython-39.pyc
    │   ├── experiment.cpython-39.pyc
    │   ├── feature_extractors.cpython-310.pyc
    │   ├── feature_extractors.cpython-311.pyc
    │   ├── feature_extractors.cpython-39.pyc
    │   ├── heads.cpython-310.pyc
    │   ├── heads.cpython-311.pyc
    │   ├── heads.cpython-39.pyc
    │   ├── helpers.cpython-39.pyc
    │   ├── loss.cpython-39.pyc
    │   ├── models.cpython-310.pyc
    │   ├── models.cpython-311.pyc
    │   ├── models.cpython-39.pyc
    │   ├── modules.cpython-310.pyc
    │   ├── modules.cpython-311.pyc
    │   ├── modules.cpython-39.pyc
    │   ├── pretrained.cpython-310.pyc
    │   ├── pretrained.cpython-311.pyc
    │   ├── pretrained.cpython-39.pyc
    │   ├── spectral_ops.cpython-310.pyc
    │   ├── spectral_ops.cpython-311.pyc
    │   └── spectral_ops.cpython-39.pyc
    ├── dataset.py
    ├── discriminators.py
    ├── experiment.py
    ├── feature_extractors.py
    ├── heads.py
    ├── helpers.py
    ├── loss.py
    ├── models.py
    ├── modules.py
    ├── pretrained.py
    └── spectral_ops.py

34 directories, 165 files