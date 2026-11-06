"""Builds notebooks/00_run_everything.ipynb: runs the project step by step from Jupyter (local or Colab)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Chạy toàn bộ dự án từ Jupyter

Mỗi ô chạy một bước. **Chạy từ trên xuống.** Các bước nhanh (vài phút) chạy mặc định; các bước rất lâu có công tắc `RUN_HEAVY` ở ô 2.

| Bước | Thời gian (máy 16 nhân) | Cần khi nào |
|---|---|---|
| Tuần 1 `main.py` | 2-5 phút (+ tải 548 MB lần đầu) | **Luôn cần** (tạo dữ liệu train/test) |
| Kiểm thử | < 1 phút | Nên chạy |
| Tuần 2 lưới thí nghiệm | ~10 phút | Chỉ khi muốn chạy lại; kết quả đã có sẵn |
| Tuần 2 xác nhận toàn train | 16-20 phút | Chỉ khi muốn chạy lại |
| Tuần 3 CV (4 giai đoạn) | 1-2 giờ | Chỉ khi muốn chạy lại |
| Sinh bảng/báo cáo | < 1 phút | Nên chạy (dùng kết quả có sẵn) |

**Không có bước nào chạy lại `final_eval`** (tập test đã được đánh giá một lần; chương trình sẽ từ chối)."""))
C.append(code("""# Ô 1: chuẩn bị. Trên Colab sẽ tự clone repo và cài thư viện; trên máy local chỉ kiểm tra thư mục.
import os, sys, subprocess
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    if not Path('Car_price_analysis').exists():
        subprocess.run(['git', 'clone', 'https://github.com/Matrixx2005/Car_price_analysis.git'], check=True)
    os.chdir('Car_price_analysis')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
elif Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)
print('Thư mục làm việc:', Path.cwd())
assert (Path.cwd() / 'main.py').exists(), 'Hãy mở notebook trong thư mục dự án (có file main.py)'
print('Python:', sys.version.split()[0], '| Colab:', IN_COLAB)"""))
C.append(code("""# Ô 2: công tắc. Đặt True CHỈ khi muốn chạy lại các thí nghiệm nặng (xem bảng thời gian ở trên).
RUN_HEAVY = False

def run(*args):
    \"\"\"Chạy `python -m ...` / `python file.py` trong cùng môi trường và in kết quả ngay khi có.\"\"\"
    cmd = [sys.executable, '-u', *args]
    print('>>', ' '.join(args))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                         env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    for line in p.stdout:
        if not line.lstrip().startswith(tuple('0123456789')) or 'tasks' in line:   # bỏ dòng thanh tiến trình tải file
            print(line, end='')
    p.wait()
    print(f'-- kết thúc, mã thoát {p.returncode}')
    return p.returncode"""))
C.append(md("## Bước 1. Tuần 1: tải dữ liệu, làm sạch, chia train/test (bắt buộc)"))
C.append(code("""run('main.py')"""))
C.append(md("Kỳ vọng dòng cuối: `Done. n_clean = 94266 | train/test = 75457 / 18809`."))
C.append(md("## Bước 2. Kiểm thử"))
C.append(code("""run('-m', 'pytest', '-q')"""))
C.append(md("Kỳ vọng: `66 passed`."))
C.append(md("## Bước 3. Xem các bảng số liệu chính (dùng khi trình bày)"))
C.append(code("""run('demo.py')"""))
C.append(md("""## Bước 4. Tuần 2 và tuần 3: sinh lại bảng, biểu đồ, báo cáo từ kết quả có sẵn (nhanh)
Các lệnh này **không** chạy lại thí nghiệm nặng; chúng đọc kết quả đã lưu trong `reports/tables/`."""))
C.append(code("""run('-m', 'src.week2_analysis')
run('-m', 'src.reporting_week2')
run('-m', 'src.week3_analysis')
run('-m', 'src.reporting_week3')"""))
C.append(md("""## Bước 5 (tuỳ chọn, RẤT LÂU). Chạy lại các thí nghiệm
Chỉ chạy khi `RUN_HEAVY = True`. Nếu một bước báo "chữ ký khác", hãy xóa file kết quả và file chữ ký tương ứng (xem `HUONG_DAN_JUPYTER.md`).
Các bước đều **chạy tiếp được**: nếu bị dừng, chạy lại đúng ô đó."""))
C.append(code("""if RUN_HEAVY:
    run('-m', 'src.experiments')              # tuần 2 lưới chính, ~10 phút
    run('-m', 'src.experiments', '--confirm') # tuần 2 xác nhận trên toàn train, 16-20 phút
    run('-m', 'src.week3_experiments', '--stage', 'all')   # tuần 3 CV, 1-2 giờ
    run('-m', 'src.week3_select')             # đóng băng cấu hình (CHỈ khi chưa có file đánh dấu test)
else:
    print('Bỏ qua (RUN_HEAVY = False). Kết quả đã có sẵn trong reports/tables/.')"""))
C.append(md("## Bước 6. Xem báo cáo"))
C.append(code("""from IPython.display import Markdown, display
display(Markdown(Path('reports/week3_report.md').read_text(encoding='utf-8')[:6000] + '\\n\\n... (xem đầy đủ trong reports/week3_report.md)'))"""))
nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/00_run_everything.ipynb")
