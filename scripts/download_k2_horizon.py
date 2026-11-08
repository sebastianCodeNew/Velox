"""
Velox Resilient Background Downloader for K2-Horizon-3.7B Weights
Target: weights/k2_horizon_q4_k_m.gguf (3.15 GB)
Features: Zero-dependency (urllib only), resume support via HTTP Range, streaming progress logging.
"""

import os
import sys
import time
import urllib.request

GGUF_URL = "https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf"
OUTPUT_FILE = os.path.join("weights", "k2_horizon_q4_k_m.gguf")
CHUNK_SIZE = 1024 * 1024  # 1 MB

def download_with_resume(url: str, dest_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    # Check existing size for resume
    initial_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Velox-Downloader/1.0")
    if initial_bytes > 0:
        req.add_header("Range", f"bytes={initial_bytes}-")
        print(f"[*] Resuming download from byte {initial_bytes:,} ({initial_bytes / (1024**3):.2f} GiB)")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            content_range = resp.headers.get("Content-Range")
            total_bytes = None
            if content_range:
                # Content-Range: bytes START-END/TOTAL
                try:
                    total_bytes = int(content_range.split("/")[-1])
                except Exception:
                    pass
            if total_bytes is None:
                cl = resp.headers.get("Content-Length")
                if cl:
                    total_bytes = int(cl) + initial_bytes
            
            print(f"[*] Total file size: {total_bytes:,} bytes ({total_bytes / (1024**3):.2f} GiB)" if total_bytes else "[*] Size unknown")

            mode = "ab" if initial_bytes > 0 and status == 206 else "wb"
            downloaded = initial_bytes if mode == "ab" else 0
            
            start_time = time.time()
            last_log = start_time
            
            with open(dest_path, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    now = time.time()
                    if now - last_log >= 2.0:
                        elapsed = now - start_time
                        speed = ((downloaded - initial_bytes) / elapsed) / (1024 * 1024) if elapsed > 0 else 0
                        pct = (downloaded / total_bytes * 100.0) if total_bytes else 0.0
                        eta = ((total_bytes - downloaded) / (speed * 1024 * 1024)) if total_bytes and speed > 0 else 0
                        print(f"[*] [{pct:5.1f}%] {downloaded / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GiB | Speed: {speed:.2f} MB/s | ETA: {eta:.0f}s", flush=True)
                        last_log = now
                        
            print(f"[+] Download complete: {dest_path} ({downloaded:,} bytes)", flush=True)
            return True
    except Exception as e:
        print(f"[-] Download error: {e}", file=sys.stderr, flush=True)
        return False

if __name__ == "__main__":
    download_with_resume(GGUF_URL, OUTPUT_FILE)
