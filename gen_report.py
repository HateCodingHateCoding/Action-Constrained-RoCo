import os, json, glob, base64

TASKS = [
    ('cabinet', '开柜取物', 'data/glm5_cabinet', 'DeepSeek V3.1'),
    ('sort', '方块分拣', 'data/glm5_sort', 'DeepSeek V3.1'),
    ('rope', '搬运绳子', 'data/glm5_rope', 'DeepSeek V3.1'),
    ('sandwich', '做三明治', 'data/glm5_sandwich', 'DeepSeek V3.1'),
    ('sweep', '协作扫地', 'data/glm5_sweep', 'DeepSeek V3.1'),
    ('relay', '流水线装配', 'data/glm5_relay', 'DeepSeek V4 Pro'),
    ('pack', '杂货打包', 'data/glm5_pack', 'GLM-4.7'),
]

def get_steps(data_dir):
    vids = sorted(glob.glob(os.path.join(data_dir, 'run_0/step_*/execute.mp4')))
    steps = []
    for vid in vids:
        step_dir = os.path.dirname(vid)
        step_num = int(os.path.basename(step_dir).split('_')[1])
        prompts = sorted(glob.glob(os.path.join(step_dir, 'prompts/replan*_feedback_*.json')))
        action_text = ""
        if prompts:
            try:
                data = json.load(open(prompts[-1]))
                for item in data:
                    if item.get('sender') == 'Action':
                        action_text = item['message']
            except:
                pass
        steps.append(dict(num=step_num, vid=vid.replace('\\', '/'), action=action_text))
    return steps

def vid_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Build HTML
html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RoCo 多机器人协作实验报告</title>
<style>
PLACEHOLDER_CSS
</style>
</head>
<body>
<div class="sidebar">
    <h2>RoCo 实验报告</h2>
PLACEHOLDER_TABS
    <div class="footer">多模型 · 智谱AI/NVIDIA<br>共 7 个任务</div>
</div>
<div class="main">
PLACEHOLDER_PANELS
</div>
<script>
PLACEHOLDER_JS
</script>
</body>
</html>'''

css = '''*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#0f0c29;color:#e0e0e0;display:flex;height:100vh;overflow:hidden}
.sidebar{width:240px;background:rgba(255,255,255,.04);border-right:1px solid rgba(255,255,255,.08);padding:16px 0;flex-shrink:0;display:flex;flex-direction:column}
.sidebar h2{padding:16px 20px;font-size:15px;background:linear-gradient(90deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:1px}
.tab{padding:12px 20px;cursor:pointer;font-size:14px;border-left:3px solid transparent;transition:all .2s}
.tab:hover{background:rgba(255,255,255,.06)}
.tab.active{background:rgba(102,126,234,.15);border-left-color:#667eea;color:#fff;font-weight:600}
.tab .model-tag{font-size:10px;color:#888;margin-left:4px}
.main{flex:1;overflow-y:auto;padding:20px 28px}
.main::-webkit-scrollbar{width:8px}
.main::-webkit-scrollbar-thumb{background:#444;border-radius:4px}
.panel{display:none}
.panel.active{display:block}
.task-header{margin-bottom:16px;padding:12px 0;border-bottom:1px solid rgba(255,255,255,.1)}
.task-header h3{font-size:20px;color:#667eea}
.task-header .meta{font-size:12px;color:#888;margin-top:4px}
.step{background:rgba(255,255,255,.05);border-radius:10px;margin-bottom:14px;border:1px solid rgba(255,255,255,.08);overflow:hidden}
.step:hover{border-color:rgba(102,126,234,.3)}
.step-hdr{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;cursor:pointer;background:rgba(255,255,255,.02);user-select:none}
.step-hdr span:first-child{font-size:16px;color:#667eea;font-weight:600}
.arrow{color:#888;font-size:12px;transition:transform .2s}
.step-body{padding:0}
.step-body.hide{display:none}
.vid-box{text-align:center;padding:16px;background:rgba(0,0,0,.2)}
.vid-box video{width:100%;max-width:720px;border-radius:8px;border:2px solid rgba(102,126,234,.25);box-shadow:0 4px 24px rgba(0,0,0,.5)}
.action-box{padding:12px 16px;font-size:13px;color:#aaa;font-family:monospace;white-space:pre-wrap;border-top:1px solid rgba(255,255,255,.06)}
.footer{padding:12px 20px;font-size:11px;color:#555;border-top:1px solid rgba(255,255,255,.06);margin-top:auto}'''

js = '''function showTask(i){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('tab-'+i).classList.add('active');
  document.getElementById('panel-'+i).classList.add('active');
}
document.querySelectorAll('.step-hdr').forEach(h=>{
  h.onclick=()=>{
    let b=h.nextElementSibling;
    b.classList.toggle('hide');
    h.querySelector('.arrow').textContent=b.classList.contains('hide')?'▶':'▼';
  }
});
showTask(0);'''

tabs = ""
panels = ""
for i, (key, name, data_dir, model) in enumerate(TASKS):
    steps = get_steps(data_dir)
    n = len(steps)
    icon = "✅" if n > 0 else "❌"
    active = " active" if i == 0 else ""
    tabs += f'    <div class="tab{active}" onclick="showTask({i})" id="tab-{i}">{icon} {name} <span class="model-tag">({model})</span></div>\n'

    panels += f'<div class="panel{"  active" if i==0 else ""}" id="panel-{i}">\n'
    panels += f'<div class="task-header"><h3>{name}</h3><div class="meta">模型: {model} | 步数: {n}</div></div>\n'

    for s in steps:
        b64 = vid_to_b64(s['vid'])
        action_html = s['action'].replace('<', '&lt;').replace('>', '&gt;') if s['action'] else '(无动作记录)'
        panels += f'''<div class="step">
<div class="step-hdr"><span>Step {s["num"]}</span><span class="arrow">▼</span></div>
<div class="step-body">
<div class="vid-box"><video controls><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video></div>
<div class="action-box">{action_html}</div>
</div></div>\n'''

    if n == 0:
        panels += '<div style="padding:40px;text-align:center;color:#666">暂无执行结果</div>\n'
    panels += '</div>\n'

html = html.replace('PLACEHOLDER_CSS', css)
html = html.replace('PLACEHOLDER_TABS', tabs)
html = html.replace('PLACEHOLDER_PANELS', panels)
html = html.replace('PLACEHOLDER_JS', js)

with open('report.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Report generated: report.html ({len(html)//1024}KB)")
