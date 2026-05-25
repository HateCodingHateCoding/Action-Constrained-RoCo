"""双击运行此脚本，自动打开浏览器查看实验报告"""
import os
import sys
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from glob import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PORT = 8090

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

def main():
    # 找到所有实验报告
    html_files = sorted(glob(os.path.join(DATA_DIR, "*/run_0/steps*.html")))
    if not html_files:
        print("没有找到实验报告！请先运行任务。")
        input("按回车退出...")
        return

    print("=" * 50)
    print("  RoCo 实验报告查看器")
    print("=" * 50)
    for i, f in enumerate(html_files):
        task = f.split(os.sep)[-3]
        fname = os.path.basename(f)
        success = "✅" if "True" in fname else "❌"
        print(f"  [{i+1}] {success} {task}")
    print(f"  [0] 全部打开")
    print()

    choice = input("输入编号查看 (直接回车=全部打开): ").strip()
    if choice == "" or choice == "0":
        selected = html_files
    else:
        try:
            selected = [html_files[int(choice) - 1]]
        except:
            selected = html_files

    os.chdir(DATA_DIR)
    server = HTTPServer(("127.0.0.1", PORT), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"\n本地服务器已启动: http://127.0.0.1:{PORT}")

    for f in selected:
        rel = os.path.relpath(f, DATA_DIR).replace("\\", "/")
        url = f"http://127.0.0.1:{PORT}/{rel}"
        webbrowser.open(url)
        print(f"  已打开: {rel}")

    print("\n按 Ctrl+C 或关闭此窗口退出服务器")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已关闭")

if __name__ == "__main__":
    main()
