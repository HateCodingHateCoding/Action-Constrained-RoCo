"""生成集成的全任务 HTML 报告，视频内嵌 base64，双击即可查看"""
import os
import json
import base64
from glob import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")

TASK_NAMES_CN = {
    "cabinet": "开柜取物",
    "sort": "方块分拣",
    "sweep": "协作扫地",
    "sandwich": "做三明治",
    "rope": "搬运绳子",
    "pack": "杂货打包",
}

ROLE_COLORS = {
    "Alice": ("#e91e63", "#fce4ec"),
    "Bob": ("#2196f3", "#e3f2fd"),
    "Chad": ("#ff9800", "#fff3e0"),
    "Dave": ("#4caf50", "#e8f5e9"),
    "Planner": ("#9c27b0", "#f3e5f5"),
    "Feedback": ("#f44336", "#ffebee"),
    "Action": ("#009688", "#e0f2f1"),
}

SENDER_KEYS = {"Alice", "Bob", "Chad", "Dave", "Planner", "Feedback", "Action"}


def encode_video(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_task_data(task_dir):
    run_dir = os.path.join(task_dir, "run_0")
    if not os.path.isdir(run_dir):
        return None

    result_files = glob(os.path.join(run_dir, "steps*_success_*.json"))
    if not result_files:
        return None
    result = json.load(open(result_files[0]))
    success = result.get("success", False)

    step_dirs = sorted(glob(os.path.join(run_dir, "step_*")))
    steps = []
    for sd in step_dirs:
        json_files = sorted(glob(os.path.join(sd, "prompts", "*.json")))
        messages = []
        for jf in json_files:
            try:
                data = json.load(open(jf, encoding="utf-8"))
            except:
                try:
                    data = json.load(open(jf))
                except:
                    continue
            if isinstance(data, dict):
                data = [data]
            for d in data:
                if d.get("sender") in SENDER_KEYS:
                    messages.append(d)

        vid_path = os.path.join(sd, "execute.mp4")
        vid_b64 = encode_video(vid_path)
        steps.append({"messages": messages, "video": vid_b64})

    task_key = os.path.basename(task_dir).replace("glm5_", "")
    return {
        "key": task_key,
        "name_cn": TASK_NAMES_CN.get(task_key, task_key),
        "success": success,
        "num_steps": len(steps),
        "steps": steps,
    }


def build_html(tasks):
    tabs_html = ""
    panels_html = ""

    for i, t in enumerate(tasks):
        active = "active" if i == 0 else ""
        icon = "✅" if t["success"] else "❌"
        tabs_html += f'<div class="tab {active}" onclick="showTask({i})" id="tab-{i}">{icon} {t["name_cn"]}</div>\n'

        steps_content = ""
        for si, step in enumerate(t["steps"]):
            vid_html = ""
            if step["video"]:
                vid_html = f'''<div class="vid-box">
                    <video controls muted loop playsinline>
                        <source src="data:video/mp4;base64,{step["video"]}" type="video/mp4">
                    </video>
                </div>'''

            msgs_html = ""
            for m in step["messages"]:
                sender = m.get("sender", "")
                text = str(m.get("message", ""))
                text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                c, bg = ROLE_COLORS.get(sender, ("#666", "#f5f5f5"))
                msgs_html += f'<div class="msg" style="border-left:4px solid {c};background:{bg}"><div class="sender" style="color:{c}">{sender}</div><div class="text">{text}</div></div>'

            steps_content += f'''
            <div class="step">
                <div class="step-hdr" onclick="this.nextElementSibling.classList.toggle('hide')">
                    <span>📋 Step {si}</span><span class="arrow">▼</span>
                </div>
                <div class="step-body">
                    {vid_html}
                    <div class="chat">{msgs_html}</div>
                </div>
            </div>'''

        display = "" if i == 0 else "display:none;"
        panels_html += f'<div class="panel" id="panel-{i}" style="{display}">{steps_content}</div>\n'

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RoCo 多机器人协作 - GLM-5.1 实验报告</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#0f0c29;color:#e0e0e0;display:flex;height:100vh;overflow:hidden}}
.sidebar{{width:220px;background:rgba(255,255,255,.04);border-right:1px solid rgba(255,255,255,.08);padding:16px 0;flex-shrink:0;display:flex;flex-direction:column}}
.sidebar h2{{padding:16px 20px;font-size:15px;background:linear-gradient(90deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:1px}}
.tab{{padding:12px 20px;cursor:pointer;font-size:14px;border-left:3px solid transparent;transition:all .2s}}
.tab:hover{{background:rgba(255,255,255,.06)}}
.tab.active{{background:rgba(102,126,234,.15);border-left-color:#667eea;color:#fff;font-weight:600}}
.main{{flex:1;overflow-y:auto;padding:20px 28px}}
.main::-webkit-scrollbar{{width:8px}}
.main::-webkit-scrollbar-thumb{{background:#444;border-radius:4px}}
.panel{{}}
.step{{background:rgba(255,255,255,.05);border-radius:10px;margin-bottom:14px;border:1px solid rgba(255,255,255,.08);overflow:hidden}}
.step:hover{{border-color:rgba(102,126,234,.3)}}
.step-hdr{{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;cursor:pointer;background:rgba(255,255,255,.02);user-select:none}}
.step-hdr span:first-child{{font-size:16px;color:#667eea;font-weight:600}}
.arrow{{color:#888;font-size:12px;transition:transform .2s}}
.step-body.hide{{display:none}}
.vid-box{{text-align:center;padding:16px;background:rgba(0,0,0,.2);border-bottom:1px solid rgba(255,255,255,.06)}}
.vid-box video{{width:100%;max-width:720px;border-radius:8px;border:2px solid rgba(102,126,234,.25);box-shadow:0 4px 24px rgba(0,0,0,.5)}}
.chat{{padding:12px 16px;max-height:420px;overflow-y:auto}}
.chat::-webkit-scrollbar{{width:5px}}
.chat::-webkit-scrollbar-thumb{{background:#555;border-radius:3px}}
.msg{{margin:6px 0;padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.65}}
.msg:hover{{filter:brightness(1.05)}}
.sender{{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}}
.text{{color:#333;word-break:break-word}}
.footer{{padding:12px 20px;font-size:11px;color:#555;border-top:1px solid rgba(255,255,255,.06);margin-top:auto}}
</style>
</head>
<body>
<div class="sidebar">
    <h2>🤖 RoCo 实验报告</h2>
    {tabs_html}
    <div class="footer">GLM-5.1 · 智谱AI<br>共 {len(tasks)} 个任务</div>
</div>
<div class="main">
    {panels_html}
</div>
<script>
function showTask(idx) {{
    document.querySelectorAll('.tab').forEach((t,i)=>{{t.classList.toggle('active',i===idx)}});
    document.querySelectorAll('.panel').forEach((p,i)=>{{p.style.display=i===idx?'':'none'}});
}}
</script>
</body>
</html>'''


def main():
    task_dirs = sorted(glob(os.path.join(DATA_DIR, "glm5_*")))
    tasks = []
    for td in task_dirs:
        print(f"处理 {os.path.basename(td)}...")
        t = build_task_data(td)
        if t:
            tasks.append(t)
            print(f"  ✅ {t['name_cn']}: {t['num_steps']} 步, {'成功' if t['success'] else '未完成'}")
        else:
            print(f"  ⏭️ 跳过（无结果）")

    if not tasks:
        print("没有找到已完成的任务！")
        return

    html = build_html(tasks)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\n✅ 报告已生成: {OUTPUT} ({size_mb:.1f} MB)")
    print("直接双击 report.html 即可在浏览器中查看！")


if __name__ == "__main__":
    main()
