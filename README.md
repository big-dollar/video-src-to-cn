# Video OCR Translate

从视频中提取英文字幕并使用大模型翻译成中文，自动生成双语/中文字幕并烧录到视频中的自动化工具。专为烹饪食谱、短语教学等密集型短句视频优化。

## 功能特性

- **高精度 OCR**: 使用 EasyOCR 从视频画面中自动识别英文字幕
- **智能排版去重**: 针对食谱短视频专门优化，自动过滤重复帧，精准切分食材短语，拒绝“长句缝合怪”
- **大模型翻译**: 使用 Gemini 大模型结合上下文批量翻译字幕为中文
- **多进程极速模式**: 支持 CPU 多核并发提取与识别，榨干硬件性能，提取速度提升数倍
- **交互式人工校对**: 翻译完成后自动暂停，允许用户人工修改 SRT 文件（如删除水印误识别），确认后再烧录
- **无缝烧录**: 自动使用 FFmpeg 生成硬字幕视频
- **网络友好**: 自动处理本地代理造成的 SSL 证书校验问题

## 环境要求

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) (必须安装并配置到系统环境变量 PATH 中，用于烧录字幕)
- NVIDIA GPU (可选，用于加速单进程模式的 OCR)

## 安装依赖

```bash
# 由于兼容性原因，强烈建议安装低于 2.x 版本的 numpy
pip install "numpy<2"

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
# 视频抽帧频率（默认每秒1帧）
FRAMES_PER_SECOND=1
# 字幕断句/去重判断间隔阈值（秒）
SUBTITLE_GAP_THRESHOLD=0.5
```

## 使用方法

提供两个脚本供选择：

### 1. 稳定版（推荐用于普通电脑或使用 GPU）
占用内存小，串行处理，如果配置了 GPU 则速度极快。
```bash
python video_ocr_translate.py <你的视频文件.mp4>
```

### 2. 多进程极速版（推荐用于纯 CPU 且核心数较多的电脑）
自动调用所有 CPU 核心并发处理，识别速度极快，但**占用内存较大**。
```bash
python video_ocr_translate_multi.py <你的视频文件.mp4>
```
*(如果不加文件名直接运行脚本，则会自动批量处理当前目录下的所有支持的视频文件)*

## 输出文件

- `xxx_output.srt` - 翻译后的外挂字幕文件 (可在中途暂停时手动修改)
- `xxx_with_subs.mp4` - 带有硬字幕的最终视频

## 注意事项

- 首次运行需要下载 EasyOCR 的轻量级模型。
- 多进程版本由于会同时加载多个 OCR 模型，请确保电脑有足够的可用内存。如果在执行时卡死，请修改脚本中的 `PROCESS_COUNT` 降低并发数。
- YouTube下载命令参考:
```bash
yt-dlp --proxy "http://127.0.0.1:10888" --cookies cookies.txt --js-runtimes node "https://www.youtube.com/watch?v=7Bjhqlr9fkc" -f "bestvideo[height>=1080]+bestaudio/best[height>=1080]/best" -o "video_%(id)s.%(ext)s"
```
