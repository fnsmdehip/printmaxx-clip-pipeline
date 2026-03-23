#!/usr/bin/env python3

from __future__ import annotations
"""
PRINTMAXX Clip Automation Pipeline — Download → Transcribe → Detect Viral → Auto-Clip

Flow:
    1. Download stream/video via yt-dlp
    2. Transcribe with whisper (local or API)
    3. Detect viral moments (keyword spikes, energy, laughter, reactions)
    4. Auto-clip with ffmpeg (adds padding, transitions)
    5. Output clips ready for posting

Usage:
    python3 AUTOMATIONS/clip_automation_pipeline.py --url "URL"                    # Full pipeline
    python3 AUTOMATIONS/clip_automation_pipeline.py --url "URL" --transcribe-only  # Just transcribe
    python3 AUTOMATIONS/clip_automation_pipeline.py --url "URL" --clips 10         # Max 10 clips
    python3 AUTOMATIONS/clip_automation_pipeline.py --batch urls.txt               # Batch process
    python3 AUTOMATIONS/clip_automation_pipeline.py --folder /path/to/videos       # Local videos
    python3 AUTOMATIONS/clip_automation_pipeline.py --status                       # Pipeline status

Dependencies:
    pip3 install yt-dlp openai-whisper
    brew install ffmpeg  (or apt install ffmpeg)
"""

import os
import sys
import json
import subprocess
import re
import math
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "output" / "clips"
TRANSCRIPT_DIR = BASE / "output" / "transcripts"
LOG_DIR = BASE / "AUTOMATIONS" / "logs"
LEDGER = BASE / "LEDGER"


def ensure_dirs():
    for d in [OUTPUT_DIR, TRANSCRIPT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """Check if required tools are installed."""
    deps = {"yt-dlp": "pip3 install yt-dlp", "ffmpeg": "brew install ffmpeg"}
    missing = []
    for tool, install in deps.items():
        try:
            # yt-dlp uses --version (rc=0), ffmpeg uses -version (rc=8)
            for flag in ["--version", "-version"]:
                result = subprocess.run([tool, flag], capture_output=True)
                if result.returncode == 0 or result.stdout or result.stderr:
                    break
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            missing.append(f"  {tool}: {install}")
    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(m)
        print("\nInstall them and retry.")
        return False
    return True


def download_video(url, output_dir=None):
    """Download video via yt-dlp. Returns path to downloaded file."""
    out = output_dir or str(OUTPUT_DIR / "raw")
    os.makedirs(out, exist_ok=True)

    template = os.path.join(out, "%(title)s_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "-o", template,
        "--no-playlist",
        "--restrict-filenames",
        url
    ]

    print(f"Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Download error: {result.stderr[:500]}")
        return None

    # Find the downloaded file - prefer the final merged mp4 on disk
    # yt-dlp stdout can reference intermediate filenames (.f614.mp4) that
    # don't exist after merging, so always prefer the actual file on disk.
    mp4s = sorted(Path(out).glob("*.mp4"), key=os.path.getmtime, reverse=True)
    if mp4s:
        return str(mp4s[0])

    # Fallback: parse stdout for path hints
    for line in result.stdout.splitlines():
        if "has already been downloaded" in line or "[download] Destination:" in line:
            match = re.search(r'(?:Destination:|already been downloaded[^"]*"?)(.+?)(?:"|$)', line)
            if match:
                p = match.group(1).strip().strip('"')
                if os.path.exists(p):
                    return p

    return None


def get_video_duration(path):
    """Get video duration in seconds."""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        return 0


def transcribe_video(video_path, model_size="base"):
    """Transcribe video using whisper. Returns segments with timestamps."""
    print(f"Transcribing: {os.path.basename(video_path)} (model: {model_size})")

    try:
        import whisper
    except ImportError:
        print("Whisper not installed. Run: pip3 install openai-whisper")
        print("Falling back to ffmpeg subtitle extraction...")
        return extract_subtitles_ffmpeg(video_path)

    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "avg_logprob": seg.get("avg_logprob", 0),
        })

    # Save transcript
    transcript_path = TRANSCRIPT_DIR / f"{Path(video_path).stem}_transcript.json"
    with open(transcript_path, "w") as f:
        json.dump({"video": video_path, "segments": segments,
                    "full_text": result.get("text", "")}, f, indent=2)

    print(f"Transcript saved: {transcript_path} ({len(segments)} segments)")
    return segments


def extract_subtitles_ffmpeg(video_path):
    """Fallback: extract embedded subtitles via ffmpeg."""
    srt_path = str(Path(video_path).with_suffix(".srt"))
    cmd = ["ffmpeg", "-i", video_path, "-map", "0:s:0", srt_path, "-y"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(srt_path):
        print("No subtitles found. Using silence detection for clipping.")
        return []

    return parse_srt(srt_path)


def parse_srt(srt_path):
    """Parse SRT file into segments."""
    segments = []
    with open(srt_path) as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            time_line = lines[1]
            text = " ".join(lines[2:])
            match = re.match(r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)", time_line)
            if match:
                start = srt_time_to_seconds(match.group(1))
                end = srt_time_to_seconds(match.group(2))
                segments.append({"start": start, "end": end, "text": text})

    return segments


def srt_time_to_seconds(t):
    """Convert SRT timestamp to seconds."""
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# --- VIRAL MOMENT DETECTION ---

VIRAL_KEYWORDS = [
    # Reaction triggers
    "oh my god", "no way", "what the", "holy", "insane", "crazy", "wait",
    "unbelievable", "i can't believe", "did you see", "look at this",
    # Emotional peaks
    "let's go", "yes", "finally", "clutch", "amazing", "incredible",
    # Controversy/drama
    "drama", "exposed", "cancelled", "beef", "fight", "argument",
    # Humor
    "lmao", "lol", "bro", "dead", "i'm done", "stop",
    # Educational hooks
    "here's the thing", "most people don't know", "secret", "trick",
    "hack", "nobody talks about", "the real reason",
    # Call to action moments
    "subscribe", "follow", "link in", "check out",
]

VIRAL_PHRASES_WEIGHTED = {
    "oh my god": 3, "no way": 3, "let's go": 2, "holy shit": 4,
    "what the fuck": 3, "i can't believe": 3, "that's insane": 3,
    "here's the thing": 2, "most people don't know": 3,
    "nobody talks about": 3, "the real reason": 2,
    "wait wait wait": 4, "oh no": 2, "clutch": 2,
}


def detect_viral_moments(segments, min_clip_duration=15, max_clip_duration=60):
    """Score segments and find viral clip-worthy moments."""
    if not segments:
        return []

    scored = []
    for i, seg in enumerate(segments):
        text_lower = seg["text"].lower()
        score = 0

        # Keyword scoring
        for kw in VIRAL_KEYWORDS:
            if kw in text_lower:
                score += VIRAL_PHRASES_WEIGHTED.get(kw, 1)

        # Exclamation/question density
        score += text_lower.count("!") * 0.5
        score += text_lower.count("?") * 0.3

        # ALL CAPS words (shouting = energy)
        caps_words = len([w for w in seg["text"].split() if w.isupper() and len(w) > 2])
        score += caps_words * 0.5

        # Short rapid segments (fast talking = energy)
        duration = seg["end"] - seg["start"]
        words = len(seg["text"].split())
        if duration > 0:
            words_per_sec = words / duration
            if words_per_sec > 3.5:  # Fast speech
                score += 1.5

        # Confidence boost (whisper confidence)
        if seg.get("avg_logprob", 0) > -0.3:
            score += 0.5

        scored.append({**seg, "viral_score": score, "index": i})

    # Sort by viral score
    scored.sort(key=lambda x: -x["viral_score"])

    # Build clip windows around high-score segments
    clips = []
    used_ranges = set()

    for seg in scored:
        if seg["viral_score"] < 1.5:
            break

        # Build a clip window around this segment
        center = (seg["start"] + seg["end"]) / 2
        clip_start = max(0, center - max_clip_duration / 2)
        clip_end = center + max_clip_duration / 2

        # Snap to segment boundaries for clean cuts
        clip_start = find_nearest_segment_boundary(segments, clip_start, "start")
        clip_end = find_nearest_segment_boundary(segments, clip_end, "end")

        # Ensure minimum duration
        if clip_end - clip_start < min_clip_duration:
            clip_end = clip_start + min_clip_duration

        # Check overlap with existing clips
        clip_range = (int(clip_start), int(clip_end))
        overlap = False
        for used in used_ranges:
            if clip_range[0] < used[1] and clip_range[1] > used[0]:
                overlap = True
                break

        if not overlap:
            # Gather all text in this window
            clip_text = " ".join(
                s["text"] for s in segments
                if s["start"] >= clip_start and s["end"] <= clip_end
            )
            clips.append({
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "duration": round(clip_end - clip_start, 2),
                "viral_score": seg["viral_score"],
                "hook_text": seg["text"][:100],
                "full_text": clip_text[:300],
                "trigger_keyword": next(
                    (kw for kw in VIRAL_KEYWORDS if kw in seg["text"].lower()), "energy"
                ),
            })
            used_ranges.add(clip_range)

    return clips


def find_nearest_segment_boundary(segments, target_time, boundary_type):
    """Find the nearest segment start/end to a target time."""
    best = target_time
    best_diff = float("inf")
    for seg in segments:
        t = seg[boundary_type]
        diff = abs(t - target_time)
        if diff < best_diff:
            best_diff = diff
            best = t
    return best if best_diff < 3 else target_time


# --- CLIP EXTRACTION ---

def extract_clip(video_path, start, end, output_path, add_padding=True):
    """Extract a clip using ffmpeg."""
    # Add 0.5s padding for clean cuts
    if add_padding:
        start = max(0, start - 0.5)
        end = end + 0.5

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_clip_vertical(video_path, start, end, output_path):
    """Extract clip and crop to 9:16 vertical for TikTok/Reels/Shorts."""
    start = max(0, start - 0.5)
    end = end + 0.5

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_pipeline(url=None, video_path=None, max_clips=10, vertical=True, whisper_model="base"):
    """Run the full clip automation pipeline."""
    ensure_dirs()

    if not check_dependencies():
        return

    # Step 1: Download or use local video
    if url:
        video_path = download_video(url)
        if not video_path:
            print("Download failed.")
            return
    elif video_path:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return
    else:
        print("Provide --url or --folder")
        return

    print(f"\nVideo: {video_path}")
    duration = get_video_duration(video_path)
    print(f"Duration: {duration:.0f}s ({duration/60:.1f}min)")

    # Step 2: Transcribe
    segments = transcribe_video(video_path, model_size=whisper_model)
    print(f"Segments: {len(segments)}")

    # Step 3: Detect viral moments
    clips = detect_viral_moments(segments)
    print(f"Viral moments found: {len(clips)}")

    if not clips:
        print("No viral moments detected. Try with longer content or lower threshold.")
        # Fallback: equal-interval clips
        if duration > 60:
            interval = duration / min(max_clips, int(duration / 30))
            clips = [{"start": i * interval, "end": (i + 1) * interval,
                       "duration": interval, "viral_score": 0,
                       "hook_text": "interval clip", "trigger_keyword": "interval"}
                      for i in range(min(max_clips, int(duration / 30)))]
            print(f"Fallback: {len(clips)} interval clips")

    # Step 4: Extract clips
    clips = clips[:max_clips]
    video_stem = Path(video_path).stem
    clip_dir = OUTPUT_DIR / video_stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    for i, clip in enumerate(clips, 1):
        suffix = "vertical" if vertical else "landscape"
        clip_path = str(clip_dir / f"clip_{i:02d}_{suffix}.mp4")

        print(f"\nClip {i}/{len(clips)}: {clip['start']:.1f}s-{clip['end']:.1f}s "
              f"(score: {clip['viral_score']:.1f}) [{clip.get('trigger_keyword', '')}]")
        print(f"  Hook: {clip.get('hook_text', '')[:80]}")

        if vertical:
            success = extract_clip_vertical(video_path, clip["start"], clip["end"], clip_path)
        else:
            success = extract_clip(video_path, clip["start"], clip["end"], clip_path)

        if success:
            extracted.append({**clip, "file": clip_path})
            print(f"  Saved: {clip_path}")
        else:
            print(f"  FAILED to extract clip")

    # Step 5: Save manifest
    manifest = {
        "source_url": url,
        "source_video": video_path,
        "duration": duration,
        "total_segments": len(segments),
        "clips_extracted": len(extracted),
        "clips": extracted,
        "timestamp": datetime.now().isoformat(),
    }
    manifest_path = clip_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  CLIP PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Source: {os.path.basename(video_path)}")
    print(f"  Duration: {duration/60:.1f} min")
    print(f"  Clips extracted: {len(extracted)}/{len(clips)}")
    print(f"  Output: {clip_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}\n")

    # Log to pipeline tracker
    log_pipeline_run(url, video_path, len(extracted), duration)

    return extracted


def run_batch(urls_file):
    """Process multiple URLs from a file."""
    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Batch processing {len(urls)} URLs\n")
    all_clips = []
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(urls)}] {url[:80]}")
        print(f"{'='*60}")
        clips = run_pipeline(url=url)
        if clips:
            all_clips.extend(clips)

    print(f"\nBatch complete: {len(all_clips)} total clips from {len(urls)} videos")


def run_folder(folder_path):
    """Process all videos in a local folder."""
    folder = Path(folder_path)
    videos = list(folder.glob("*.mp4")) + list(folder.glob("*.mkv")) + list(folder.glob("*.webm"))
    print(f"Found {len(videos)} videos in {folder}\n")

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video.name}")
        run_pipeline(video_path=str(video))


def log_pipeline_run(url, video_path, clip_count, duration):
    """Log pipeline run for tracking."""
    log_file = LOG_DIR / "clip_pipeline.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "video": str(video_path),
        "clips": clip_count,
        "duration_sec": duration,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def print_status():
    """Print pipeline status."""
    print(f"\n{'='*50}")
    print(f"  CLIP PIPELINE STATUS")
    print(f"{'='*50}")

    # Count clips
    if OUTPUT_DIR.exists():
        clip_count = len(list(OUTPUT_DIR.rglob("*.mp4")))
        manifest_count = len(list(OUTPUT_DIR.rglob("manifest.json")))
    else:
        clip_count = 0
        manifest_count = 0

    # Count transcripts
    if TRANSCRIPT_DIR.exists():
        transcript_count = len(list(TRANSCRIPT_DIR.glob("*.json")))
    else:
        transcript_count = 0

    # Pipeline log
    log_file = LOG_DIR / "clip_pipeline.jsonl"
    runs = 0
    if log_file.exists():
        with open(log_file) as f:
            runs = sum(1 for _ in f)

    print(f"  Total clips: {clip_count}")
    print(f"  Videos processed: {manifest_count}")
    print(f"  Transcripts: {transcript_count}")
    print(f"  Pipeline runs: {runs}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Transcript dir: {TRANSCRIPT_DIR}")

    # Dependencies
    print(f"\n  Dependencies:")
    for tool in ["yt-dlp", "ffmpeg", "ffprobe"]:
        try:
            for flag in ["--version", "-version"]:
                result = subprocess.run([tool, flag], capture_output=True)
                if result.returncode == 0 or result.stdout or result.stderr:
                    print(f"    {tool}: installed")
                    break
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"    {tool}: MISSING")

    try:
        import whisper
        print(f"    whisper: installed")
    except ImportError:
        print(f"    whisper: MISSING (pip3 install openai-whisper)")

    print(f"{'='*50}\n")


def main():
    args = sys.argv[1:]

    if "--status" in args:
        print_status()
    elif "--url" in args:
        idx = args.index("--url")
        url = args[idx + 1] if idx + 1 < len(args) else ""
        max_clips = 10
        if "--clips" in args:
            ci = args.index("--clips")
            max_clips = int(args[ci + 1]) if ci + 1 < len(args) else 10
        transcribe_only = "--transcribe-only" in args
        landscape = "--landscape" in args

        if transcribe_only:
            video_path = download_video(url)
            if video_path:
                transcribe_video(video_path)
        else:
            run_pipeline(url=url, max_clips=max_clips, vertical=not landscape)

    elif "--batch" in args:
        idx = args.index("--batch")
        urls_file = args[idx + 1] if idx + 1 < len(args) else ""
        run_batch(urls_file)

    elif "--folder" in args:
        idx = args.index("--folder")
        folder = args[idx + 1] if idx + 1 < len(args) else ""
        run_folder(folder)

    else:
        print("""
PRINTMAXX Clip Automation Pipeline

Usage:
    python3 clip_automation_pipeline.py --url "URL"                    # Full pipeline
    python3 clip_automation_pipeline.py --url "URL" --transcribe-only  # Just transcribe
    python3 clip_automation_pipeline.py --url "URL" --clips 10         # Max 10 clips
    python3 clip_automation_pipeline.py --url "URL" --landscape        # Keep landscape
    python3 clip_automation_pipeline.py --batch urls.txt               # Batch process
    python3 clip_automation_pipeline.py --folder /path/to/videos       # Local folder
    python3 clip_automation_pipeline.py --status                       # Pipeline status

Dependencies:
    pip3 install yt-dlp openai-whisper
    brew install ffmpeg
""")


if __name__ == "__main__":
    main()
