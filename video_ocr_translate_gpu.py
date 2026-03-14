import os
import cv2
import easyocr
import srt
import time
import ssl
from datetime import timedelta
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 忽略 SSL 证书验证错误
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# ----------------- 配置区 -----------------
VIDEO_DIR = "." 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
FRAMES_PER_SECOND = int(os.getenv("FRAMES_PER_SECOND", "1"))
SUBTITLE_GAP_THRESHOLD = float(os.getenv("SUBTITLE_GAP_THRESHOLD", "0.5"))
# ------------------------------------------

def extract_text_from_video_gpu(video_path):
    """
    单进程 GPU 极速提取模式
    利用 GPU 强大的张量计算能力进行逐帧 OCR
    """
    start_time = time.time()
    
    # 强制检查 GPU 状态
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            device_name = torch.cuda.get_device_name(0)
            print(f"\n[🚀 硬件加速开启] 成功检测到 NVIDIA GPU: {device_name}")
            print(f"当前 PyTorch CUDA 版本: {torch.version.cuda}")
        else:
            print("\n[⚠️ 警告] 未检测到可用的 GPU。目前 PyTorch 只能使用 CPU 运行。")
            print("如果你有 NVIDIA 显卡，请参考文档重新安装 CUDA 版本的 PyTorch。")
    except ImportError:
        use_gpu = False
        print("\n[⚠️ 警告] 未安装 PyTorch，只能使用 CPU 运行。")
    
    print(f"\n正在初始化 EasyOCR 模型并加载到显存中...")
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    print("模型加载完成！")
    
    print(f"正在分析视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"视频时长: {duration:.2f} 秒, 帧率: {fps:.2f} fps")
    
    frame_skip = max(1, int(fps / FRAMES_PER_SECOND))
    
    from difflib import SequenceMatcher
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    segments = []
    current_segment = None
    frame_count = 0
    processed_count = 0
    total_tasks = total_frames // frame_skip

    print(f"开始使用 {'GPU' if use_gpu else 'CPU'} 逐帧识别字幕，预计处理 {total_tasks} 帧...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            current_time = frame_count / fps
            
            # 使用 GPU 进行推理
            # 为了进一步压榨显卡性能，可以考虑传入 batch_size (需特定版本 EasyOCR 支持)，这里先用标准方法
            results = reader.readtext(frame)
            frame_text = " ".join([text for (_, text, prob) in results if prob > 0.4]).strip()
            
            if frame_text:
                if current_segment is None:
                    current_segment = {
                        "text": frame_text,
                        "start": current_time,
                        "end": current_time + (1.0 / FRAMES_PER_SECOND)
                    }
                else:
                    if similar(current_segment["text"].lower(), frame_text.lower()) > 0.8:
                        current_segment["end"] = current_time + (1.0 / FRAMES_PER_SECOND)
                        if len(frame_text) > len(current_segment["text"]):
                            current_segment["text"] = frame_text
                    else:
                        if len(current_segment["text"]) > 2:
                            segments.append(current_segment)
                        current_segment = {
                            "text": frame_text,
                            "start": current_time,
                            "end": current_time + (1.0 / FRAMES_PER_SECOND)
                        }
            else:
                if current_segment is not None:
                    if len(current_segment["text"]) > 2:
                        segments.append(current_segment)
                    current_segment = None
                    
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"OCR 进度: {processed_count}/{total_tasks} 帧 - 耗时: {(time.time()-start_time):.2f}s")
                
        frame_count += 1
        
    if current_segment is not None and len(current_segment["text"]) > 2:
        segments.append(current_segment)
        
    cap.release()
    total_time = time.time() - start_time
    print(f"识别完成！共提取到 {len(segments)} 条初步字幕。总耗时: {total_time:.2f} 秒。")
    return segments

def translate_texts(texts, api_key, api_url, model_name):
    if not texts:
        return []
        
    print(f"正在使用 Gemini 翻译 {len(texts)} 条字幕...")
    client = genai.Client(
        api_key=api_key,
        http_options={"base_url": api_url}
    )
    
    batch_size = 50
    all_translated_texts = []
    separator = " |---| "
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"正在翻译第 {i+1} 到 {min(i+batch_size, len(texts))} 条...")
        
        combined_text = separator.join(batch_texts)
        
        prompt = f"""
        你是一个专业的影视字幕翻译员。请将以下用 '{separator}' 分隔的英文句子翻译成通顺的中文。
        注意：
        1. 保持原有句子的数量和顺序绝对一致。
        2. 翻译结果同样使用 '{separator}' 分隔。
        3. 只需要输出翻译结果，不要输出任何其他解释性文字。
        4. 视频中提取的英文可能是残缺的或包含错别字，请根据上下文推理修复并翻译成通顺的中文。
        
        待翻译文本：
        {combined_text}
        """
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                )
            )
            
            translated_combined = response.text.strip()
            translated_batch = [t.strip() for t in translated_combined.split(separator)]
            
            if len(translated_batch) == len(batch_texts):
                all_translated_texts.extend(translated_batch)
            else:
                print(f"警告：批次 {i} 翻译条数不匹配，尝试逐条翻译...")
                for text in batch_texts:
                    try:
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=f"翻译成中文，只输出结果：{text}"
                        )
                        all_translated_texts.append(resp.text.strip())
                    except Exception as e:
                        print(f"逐条翻译失败: {text} - 错误: {e}")
                        all_translated_texts.append(text)
        except Exception as e:
            print(f"批量翻译请求失败: {e}")
            all_translated_texts.extend(batch_texts)
             
    return all_translated_texts

def merge_short_subtitles(segments, min_duration=2.0, gap_threshold=0.5):
    if not segments:
        return segments
    merged = []
    current = segments[0].copy()
    
    from difflib import SequenceMatcher
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
        
    for i in range(1, len(segments)):
        next_seg = segments[i]
        gap = next_seg['start'] - current['end']
        
        text_a = current['text'].lower()
        text_b = next_seg['text'].lower()
        is_same_content = (similar(text_a, text_b) > 0.8) or (text_a in text_b) or (text_b in text_a)
        
        if gap <= gap_threshold and is_same_content:
            current['end'] = max(current['end'], next_seg['end'])
            if len(next_seg['text']) > len(current['text']):
                current['text'] = next_seg['text']
        else:
            merged.append(current)
            current = next_seg.copy()
    merged.append(current)
    return merged

def create_srt(segments, translated_texts, output_path="output.srt"):
    print("正在生成字幕文件...")
    subs = []
    valid_count = 0
    for i, (segment, trans_text) in enumerate(zip(segments, translated_texts)):
        if not trans_text.strip():
            continue
        valid_count += 1
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        sub = srt.Subtitle(
            index=valid_count, 
            start=start_time, 
            end=end_time, 
            content=trans_text
        )
        subs.append(sub)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))

def burn_subtitles_to_video(video_path, srt_path, output_video_path="output_with_subs.mp4"):
    print(f"正在将字幕烧录到视频中: {output_video_path} ...")
    import subprocess
    # 修复文件名中的特殊字符（如等号、空格等），防止 ffmpeg 参数解析错误
    # ffmpeg 的 subtitles 滤镜对路径中的特殊字符比较敏感，需要转义
    srt_path_esc = srt_path.replace('\\', '/').replace("'", "\\'")
    video_path_esc = video_path.replace('\\', '/').replace("'", "\\'")
    
    # 在 Windows 上，如果路径包含冒号（如驱动器盘符），需要特殊处理
    # 这里我们使用正则替换替换成转义字符
    import re
    srt_path_esc = re.sub(r":(?![/\\])", r"\\:", srt_path_esc)
    
    cmd = [
        'ffmpeg', '-y', '-i', video_path_esc, 
        '-vf', f"subtitles='{srt_path_esc}'", 
        '-c:a', 'copy', output_video_path
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg Error: {result.stderr}")
    else:
        print("视频处理完成！")
    print("视频处理完成！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GPU加速单进程视频OCR字幕提取工具')
    parser.add_argument('video', nargs='?', default=None, help='指定视频文件路径')
    args = parser.parse_args()
    
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"视频文件不存在: {args.video}")
            exit(1)
        video_files = [args.video]
    else:
        video_files = []
        for f in os.listdir(VIDEO_DIR):
            ext = os.path.splitext(f)[1].lower()
            if ext in video_extensions:
                video_files.append(os.path.join(VIDEO_DIR, f))
        if not video_files:
            print(f"在目录 {VIDEO_DIR} 中未找到视频文件")
            exit(1)
            
    print(f"找到 {len(video_files)} 个视频文件: {video_files}")
    
    for video_path in video_files:
        print(f"\n{'='*50}")
        print(f"正在处理: {video_path}")
        print('='*50)
        
        output_srt = os.path.splitext(video_path)[0] + "_output.srt"
        
        try:
            # 核心替换：使用 GPU 单进程版的方法
            segments = extract_text_from_video_gpu(video_path)
            
            if not segments:
                print(f"未在视频 {video_path} 中检测到任何英文字幕，跳过。")
                continue
            
            segments = merge_short_subtitles(segments, min_duration=2.0, gap_threshold=SUBTITLE_GAP_THRESHOLD)
            original_texts = [s['text'].strip() for s in segments]
            
            if not GEMINI_API_KEY:
                 print("请在 .env 配置中填入真实的 GEMINI_API_KEY")
                 exit(1)
                 
            translated_texts = translate_texts(original_texts, GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL)
            create_srt(segments, translated_texts, output_srt)
            print(f"字幕已保存至: {output_srt} (共 {len(segments)} 条)")
            
            print(f"\n{'='*50}")
            print(f"⚠️ 请现在打开并检查生成的字幕文件: {output_srt}")
            print("你可以在里面删除不需要的水印翻译，或者修正翻译错误。")
            print("修改完成后请务必保存文件！")
            print(f"{'='*50}\n")
            input("修改保存后，请按【回车键】继续进行视频烧录...")
            
            output_video = os.path.splitext(video_path)[0] + "_with_subs.mp4"
            burn_subtitles_to_video(video_path, output_srt, output_video)
            
        except Exception as e:
            print(f"处理 {video_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n所有视频处理完成！")
