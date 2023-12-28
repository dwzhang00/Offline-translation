# Offline_translation

## 介绍

本项目基于transformer模型，使用聆思科技开源的AI生态工具链LNN(ListenAI Neural Network)，完成中英翻译任务的训练、量化、仿真调试等一系列步骤，并实现在聆思CSK6-MIX芯片上进行推理。

## 环境配置

训练环境配置：https://github.com/LISTENAI/linger/blob/main/doc/tutorial/install.md

推理环境配置：https://github.com/LISTENAI/thinker/blob/main/thinker/docs/tutorial/install.md

## 主要流程

### 1. 浮点训练

```bash
conda activate linger-env
```

```bash
python run.py
```

### 2. 量化训练和导出

```bash
python run.py
```

```bash
python model_trans.py
```

### 3. 模型分析和打包

```bash
conda activate thinker-env
```

```bash
tpacker -g encoder.onnx -d True -c en_len=32 -o encoder.bin
```

```bash
tpacker -g decoder.onnx -d True -c de_len=32,memory_len=32 -o decoder.bin
```

### 4. 推理执行

```bash
bash scripts/x86_linux.sh
```

```bash
bin/test_thinker demo/test_transformer/input.txt demo/test_transformer/encoder.bin demo/test_transformer/decoder.bin demo/test_transformer/output.bin
```

### 5. 编译和烧录

## 代码参考

https://github.com/hinesboy/transformer-simple

