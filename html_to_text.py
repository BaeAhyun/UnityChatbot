import os
from pathlib import Path
from bs4 import BeautifulSoup

INPUT_DIR = "html_files"
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_html_to_text(html_file, output_file):
    """HTML 파일을 텍스트로 변환"""
    try:
        print(f"Converting {html_file.name}...")
        
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # 스크립트와 스타일 제거
        for script in soup(['script', 'style']):
            script.decompose()
        
        # 텍스트 추출
        text = soup.get_text(separator='\n', strip=True)
        
        # 빈 줄 정리
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        char_count = len(text)
        print(f"✅ Saved: {output_file.name} ({char_count:,} chars)")
        return True
    
    except Exception as e:
        print(f"❌ Error converting {html_file.name}: {e}")
        return False

def main():
    html_files = list(Path(INPUT_DIR).glob("*.html"))
    
    if not html_files:
        print(f"⚠️ No HTML files found in {INPUT_DIR}/")
        return
    
    print(f"Found {len(html_files)} HTML files\n")
    
    success = 0
    for html_file in sorted(html_files):
        output_file = Path(OUTPUT_DIR) / f"{html_file.stem}.txt"
        if convert_html_to_text(html_file, output_file):
            success += 1
    
    print(f"\n✅ Converted {success}/{len(html_files)} files")
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()