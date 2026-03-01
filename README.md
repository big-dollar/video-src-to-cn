# Video OCR Translate

从视频中提取英文字幕并翻译成中文的工具。

## 功能特性

- 使用 EasyOCR 从视频画面中自动识别英文字幕
- 使用 Gemini 大模型翻译字幕为中文
- 自动合并时间相近的短字幕，形成更完整的句子
- 生成 SRT 格式的字幕文件
- 支持将字幕烧录到视频中（硬字幕）

## 环境要求

- Python 3.8+
- FFmpeg (用于烧录字幕)
- NVIDIA GPU (可选，用于加速 OCR)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 复制 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API 配置：
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_API_URL=https://generativelanguage.googleapis.com/
GEMINI_MODEL=gemini-2.0-flash
FRAMES_PER_SECOND=1
```

## 使用方法

### 处理单个视频
```bash
python video_ocr_translate.py video.mp4
```

### 处理当前目录下所有视频
```bash
python video_ocr_translate.py
```

## 输出文件

- `xxx_output.srt` - 翻译后的字幕文件
- `xxx_with_subs.mp4` - 带有硬字幕的视频

## 注意事项

- 首次运行需要下载 EasyOCR 模型
- 使用 GPU 可大幅提升 OCR 速度
- 翻译质量取决于 API 的响应效果
