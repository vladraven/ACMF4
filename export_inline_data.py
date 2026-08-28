import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

json_path = Path("web/data/synthetic_tests.json")
if not json_path.exists():
    print("[!] Сначала запустите: python run_synthetic_tests.py")
    sys.exit(1)

raw_json = json_path.read_text(encoding="utf-8")
js_content = f"window.ACMF_DATA = {raw_json};\n"

out_js = Path("web/data.js")
out_js.write_text(js_content, encoding="utf-8")
print(f"[✓] Данные экспортированы в {out_js}")
print("[✓] Теперь просто откройте web/index.html двойным кликом!")