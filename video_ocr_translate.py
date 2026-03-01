import os
import cv2
import easyocr
import srt
from datetime import timedelta
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# ----------------- 配置区 -----------------
# 视频目录（当前目录）
VIDEO_DIR = "." 
# Gemini API Key (从 .env 读取)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Gemini API 地址 (从 .env 读取)
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/")
# 模型名称 (从 .env 读取)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# 识别频率：每秒识别多少帧 (默认1次)
FRAMES_PER_SECOND = int(os.getenv("FRAMES_PER_SECOND", "1"))
# ------------------------------------------

def extract_text_from_video(video_path):
    """使用 OCR 从视频画面中提取英文文本"""
    # 自动检测是否可以使用GPU
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("检测到 NVIDIA GPU，正在使用 GPU 加速...")
        else:
            print("未检测到 GPU，使用 CPU 运行")
    except:
        use_gpu = False
        print("使用 CPU 运行")
    
    print(f"正在初始化 EasyOCR (首次运行会下载模型)...")
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    
    print(f"正在分析视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"视频时长: {duration:.2f} 秒, 帧率: {fps:.2f} fps")
    
    # 计算需要跳过的帧数
    frame_skip = int(fps / FRAMES_PER_SECOND)
    if frame_skip == 0:
        frame_skip = 1
        
    segments = []
    current_segment = None
    frame_count = 0
    
    # 相似文本判断阈值
    similarity_threshold = 0.8 
    
    from difflib import SequenceMatcher
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    print("开始逐帧识别字幕 (可能需要较长时间)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            current_time = frame_count / fps
            
            # 为了提高OCR速度和准确度，通常字幕在视频下方
            # 如果知道字幕的大致位置，可以裁剪画面，例如只取下半部分
            # height, width, _ = frame.shape
            # cropped_frame = frame[int(height*0.7):height, 0:width]
            # results = reader.readtext(cropped_frame)
            
            # 全屏识别
            results = reader.readtext(frame)
            
            # 合并识别到的所有文本
            frame_text = " ".join([text for (_, text, prob) in results if prob > 0.4])
            frame_text = frame_text.strip()
            
            if frame_text:
                if current_segment is None:
                    # 发现新字幕
                    current_segment = {
                        "text": frame_text,
                        "start": current_time,
                        "end": current_time + (1.0 / FRAMES_PER_SECOND) # 暂定结束时间
                    }
                else:
                    # 判断当前画面文字是否与上一个字幕相同
                    if similar(current_segment["text"].lower(), frame_text.lower()) > similarity_threshold:
                        # 文本相同，延长上一个字幕的结束时间
                        current_segment["end"] = current_time + (1.0 / FRAMES_PER_SECOND)
                        # 如果新识别的文本更长(可能之前识别不全)，更新文本
                        if len(frame_text) > len(current_segment["text"]):
                            current_segment["text"] = frame_text
                    else:
                        # 文本不同，保存上一个字幕，开始新的字幕
                        if len(current_segment["text"]) > 2: # 过滤太短的杂音字符
                            segments.append(current_segment)
                        current_segment = {
                            "text": frame_text,
                            "start": current_time,
                            "end": current_time + (1.0 / FRAMES_PER_SECOND)
                        }
            else:
                # 当前帧没有文本，结束上一个字幕
                if current_segment is not None:
                    if len(current_segment["text"]) > 2:
                        segments.append(current_segment)
                    current_segment = None
                    
            if frame_count % (fps * 10) < frame_skip: # 每 10 秒打印一次进度
                print(f"进度: {current_time:.2f}s / {duration:.2f}s")
                
        frame_count += 1
        
    # 处理最后一个字幕
    if current_segment is not None and len(current_segment["text"]) > 2:
        segments.append(current_segment)
        
    cap.release()
    print(f"识别完成，共提取到 {len(segments)} 条字幕。")
    return segments

def translate_texts(texts, api_key, api_url, model_name):
    """使用 Gemini 大模型批量翻译文本"""
    if not texts:
        return []
        
    print(f"正在使用 Gemini 翻译 {len(texts)} 条字幕...")
    client = genai.Client(
        api_key=api_key,
        http_options={"base_url": api_url}
    )
    
    # 限制每次请求翻译的条数，避免超出模型上下文限制
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
                print(f"警告：批次 {i} 翻译条数不匹配 (输入 {len(batch_texts)}, 输出 {len(translated_batch)})，尝试逐条翻译...")
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
            # 如果批量失败，整个批次保留原文
            all_translated_texts.extend(batch_texts)
             
    return all_translated_texts

def merge_short_subtitles(segments, min_duration=2.0, gap_threshold=0.5):
    """合并时间相近的短字幕，形成更完整的句子"""
    if not segments:
        return segments
        
    merged = []
    current = segments[0].copy()
    
    for i in range(1, len(segments)):
        next_seg = segments[i]
        gap = next_seg['start'] - current['end']
        
        if gap <= gap_threshold:
            current['end'] = next_seg['end']
            current['text'] = current['text'] + ' ' + next_seg['text']
        else:
            merged.append(current)
            current = next_seg.copy()
    
    merged.append(current)
    return merged

def create_srt(segments, translated_texts, output_path="output.srt"):
    """生成 SRT 格式的字幕文件"""
    print("正在生成字幕文件...")
    subs = []
    # 过滤掉无法翻译的空文本
    valid_count = 0
    for i, (segment, trans_text) in enumerate(zip(segments, translated_texts)):
        # 只保留有实际文本的字幕
        if not trans_text.strip():
            continue
            
        valid_count += 1
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        
        # 可选双语输出，这里默认只输出中文
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
    """将字幕烧录到视频画面中（硬字幕）"""
    print(f"正在将字幕烧录到视频中: {output_video_path} ...")
    import subprocess
    # 注意：Windows 下需要对路径中的冒号等进行转义，或者使用正斜杠
    srt_path_esc = srt_path.replace('\\', '/').replace(':', '\\:')
    
    cmd = [
        'ffmpeg',
        '-y',               # 覆盖输出
        '-i', video_path,   # 输入视频
        '-vf', f"subtitles='{srt_path_esc}'", # 视频滤镜添加字幕
        '-c:a', 'copy',     # 音频直接复制
        output_video_path
    ]
    subprocess.run(cmd)
    print("视频处理完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='视频OCR字幕提取翻译工具')
    parser.add_argument('video', nargs='?', default=None, help='指定视频文件路径（可选，不指定则处理当前目录下所有视频）')
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
            # 1. 从视频画面中提取文字 (OCR)
            segments = extract_text_from_video(video_path)
            
            if not segments:
                print(f"未在视频 {video_path} 中检测到任何英文字幕，跳过。")
                continue
            
            # 2. 先合并时间相近的短字幕（在翻译之前合并）
            segments = merge_short_subtitles(segments, min_duration=2.0, gap_threshold=0.5)
            
            # 3. 从合并后的字幕提取文本进行翻译
            original_texts = [s['text'].strip() for s in segments]
            
            # 4. 翻译
            if not GEMINI_API_KEY:
                 print("请在代码中填入真实的 GEMINI_API_KEY")
                 exit(1)
                 
            translated_texts = translate_texts(original_texts, GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL)
            
            # 5. 生成字幕文件
            create_srt(segments, translated_texts, output_srt)
            
            print(f"字幕已保存至: {output_srt} (共 {len(segments)} 条)")
            
            # 5. 烧录到视频
            output_video = os.path.splitext(video_path)[0] + "_with_subs.mp4"
            burn_subtitles_to_video(video_path, output_srt, output_video)
            print(f"带字幕的视频已保存至: {output_video}")
            
        except Exception as e:
            print(f"处理 {video_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n所有视频处理完成！")
