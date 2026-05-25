import os
import json
from glob import glob
from os.path import join
from natsort import natsorted


ROLE_COLORS = {
    "Alice": "#e91e63",
    "Bob": "#2196f3",
    "Chad": "#ff9800",
    "Dave": "#4caf50",
    "Planner": "#9c27b0",
    "Feedback": "#f44336",
    "Action": "#009688",
    "SystemPrompt": "#607d8b",
    "UserPrompt": "#795548",
}

ROLE_BG = {
    "Alice": "#fce4ec",
    "Bob": "#e3f2fd",
    "Chad": "#fff3e0",
    "Dave": "#e8f5e9",
    "Planner": "#f3e5f5",
    "Feedback": "#ffebee",
    "Action": "#e0f2f1",
    "SystemPrompt": "#eceff1",
    "UserPrompt": "#efebe9",
}


def save_episode_html(
    episode_path,
    html_fname="display",
    video_fname="execute.mp4",
    video_include_steps=False,
    sender_keys=["Alice", "Bob", "Chad", "Dave", "Planner", "Feedback", "Action"],
):
    step_dirs = natsorted(glob(os.path.join(episode_path, "step_*")))
    if len(step_dirs) == 0:
        print("No steps found in episode path")
        return

    steps_html = []
    for step_idx, step_dir in enumerate(step_dirs):
        json_files = natsorted(glob(os.path.join(step_dir, "prompts", "*.json")))
        messages = []
        for jf in json_files:
            try:
                data = json.load(open(jf, "r", encoding="utf-8"))
            except:
                data = json.load(open(jf, "r"))
            if isinstance(data, dict):
                data = [data]
            for d in data:
                sender = d.get("sender", None)
                if sender in sender_keys:
                    messages.append(d)

        chat_html = ""
        for msg in messages:
            sender = msg.get("sender", "Unknown")
            text = msg.get("message", "")
            if isinstance(text, dict):
                text = json.dumps(text, indent=2, ensure_ascii=False)
            text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            color = ROLE_COLORS.get(sender, "#666")
            bg = ROLE_BG.get(sender, "#f5f5f5")
            chat_html += f'''
            <div class="msg" style="border-left: 4px solid {color}; background: {bg};">
                <div class="sender" style="color: {color};">{sender}</div>
                <div class="text">{text}</div>
            </div>'''

        step_name = os.path.basename(step_dir)
        vid_path = os.path.join(step_dir, video_fname)
        video_html = ""
        if os.path.exists(vid_path):
            rel_path = step_name + "/" + video_fname
            video_html = f'''
            <div class="video-box">
                <video controls autoplay muted loop playsinline>
                    <source src="{rel_path}" type="video/mp4">
                    浏览器不支持视频播放
                </video>
            </div>'''

        steps_html.append(f'''
        <div class="step-card">
            <div class="step-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <h2>📋 Step {step_idx}</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="step-body">
                {video_html}
                <div class="chat-panel">{chat_html}</div>
            </div>
        </div>''')

    task_name = os.path.basename(os.path.dirname(episode_path))
    run_name = os.path.basename(episode_path)
    success = "True" in html_fname

    full_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoCo - {task_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 20px;
}}
.header {{
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 24px;
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    backdrop-filter: blur(10px);
}}
.header h1 {{
    font-size: 28px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}
.header .meta {{ color: #aaa; font-size: 14px; }}
.badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 8px;
}}
.badge.success {{ background: #1b5e20; color: #a5d6a7; }}
.badge.fail {{ background: #b71c1c; color: #ef9a9a; }}
.step-card {{
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    margin-bottom: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}}
.step-card:hover {{ border-color: rgba(102,126,234,0.4); }}
.step-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    cursor: pointer;
    background: rgba(255,255,255,0.03);
    user-select: none;
}}
.step-header h2 {{ font-size: 18px; color: #667eea; }}
.toggle-icon {{ font-size: 14px; color: #888; transition: transform 0.3s; }}
.collapsed .toggle-icon {{ transform: rotate(-90deg); }}
.collapsed .step-body {{ display: none; }}
.step-body {{ padding: 16px 20px; }}
.video-box {{
    margin-bottom: 16px;
    text-align: center;
}}
.video-box video {{
    width: 100%;
    max-width: 800px;
    border-radius: 10px;
    border: 2px solid rgba(102,126,234,0.3);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}}
.chat-panel {{
    max-height: 500px;
    overflow-y: auto;
    padding-right: 8px;
}}
.chat-panel::-webkit-scrollbar {{ width: 6px; }}
.chat-panel::-webkit-scrollbar-thumb {{ background: #555; border-radius: 3px; }}
.msg {{
    margin: 8px 0;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.7;
}}
.msg:hover {{ filter: brightness(1.05); }}
.sender {{
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}}
.text {{ color: #333; word-break: break-word; }}
</style>
</head>
<body>
<div class="header">
    <h1>🤖 RoCo 多机器人协作实验报告</h1>
    <div class="meta">任务: {task_name} &nbsp;|&nbsp; {run_name} &nbsp;|&nbsp; 总步数: {len(step_dirs)}</div>
    <div class="badge {"success" if success else "fail"}">{"✅ 任务成功" if success else "❌ 任务未完成"}</div>
</div>
{"".join(steps_html)}
</body>
</html>'''

    out_path = os.path.join(episode_path, f"{html_fname}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved episode html to {out_path}")


def save_qa_data_html(dataset_dir, html_fname="qa_display", sender_keys=None):
    if sender_keys is None:
        sender_keys = ["Alice", "Bob", "Chad", "Dave", "SystemPrompt", "UserPrompt", "Solution", "Feedback", "Action", "Response"]
    data_dirs = natsorted(glob(os.path.join(dataset_dir, "*")))
    for data_dir in data_dirs:
        json_files = natsorted(glob(os.path.join(data_dir, "question*.json")))
        print(f"Found {len(json_files)} questions in {data_dir}")
