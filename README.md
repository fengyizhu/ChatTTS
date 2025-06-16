ChatTTS VLLM & API
========================
感谢2noise开源项目提供的ChatTTS模型和推理框架

基于ChatTTS (https://github.com/2noise/ChatTTS) 改进的推理框架，具备流式实时语音合成能力。

近期更新更多特性，欢迎【star】优先回复 
## 特性：
- 经过长时间推理验证，稳定性强
- 支持openai标准规范的语音合成接口
- 扩展流式和非流式语音合成方法
- 最快推理RTF约为0.15
- 支持并行多路同时合成，发挥GPU性能
- 优化多句话合成音色不稳定的问题。
- 无缝兼容1-2000号音色抽卡
- 流式首包延迟低至100ms

## Todo
- 保持多次合成时同音色稳定性
- 扩展音色克隆
- 集成各种数字、符号的发音

## 快速体验
感谢bella开源项目（https://github.com/LianjiaTech/bella-openapi) 提供资源试点 https://api.bella.top/playground 语音合成（模型选择chat-tts）或者实时语音对话（选择bella-realtime模型）
## 演示
https://github.com/user-attachments/assets/491ff524-e0f4-44dc-b654-59d2b5b89c81
## 要求
当前只适配了Nvidia GPU，测试验证RTX8000、V100、A系列、H系列能够稳定运行。
## 安装方式
```bash
git clone https://github.com/fengyizhu/ChatTTS
cd ChatTTS
```
安装依赖
```bash
pip install --upgrade -r requirements.txt
```
运行(参数及默认值如下，按需调整)
```bash
python -m examples/api/openai.py --host 0.0.0.0 --port 8080 --gpu_memory_utilization 0.9
```


同步接口用例
```bash
curl -X POST "http://localhost:8080/v1/audio/speech" \
-H "Content-Type: application/json" \
-d '{
    "model": "Chat-TTS",
    "input": "你好，今天天气怎么样。",
    "voice": "28",
    "speed": 1,
    "response_format": "wav",
    "stream":false
}'
```
流式接口用例
```bash
curl -X POST "http://localhost:8080/v1/audio/speech" \
-H "Content-Type: application/json" \
-d '{
    "model": "Chat-TTS",
    "input": "你好，今天天气怎么样。",
    "voice": "28",
    "speed": 1,
    "response_format": "pcm",
    "stream":true
}'
```

