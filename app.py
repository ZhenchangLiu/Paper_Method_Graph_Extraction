import streamlit as st
import json
import time
import datetime
import tempfile
from openai import OpenAI
from PyPDF2 import PdfReader
import networkx as nx
from pyvis.network import Network

# ========== Streamlit 页面设置 ==========
st.set_page_config(page_title="Paper Method Extractor", layout="wide")
st.title("Paper Method Extractor & Visualizer")

# ========== 用户输入区域 ==========
st.sidebar.header("输入参数")
api_key = st.sidebar.text_input("🔑 输入你的 OpenRouter API Key", type="password")
pdf_file = st.sidebar.file_uploader("📤 上传论文 PDF 文件", type=["pdf"])
model = st.sidebar.selectbox("选择模型", ["google/gemini-2.5-pro", "openai/gpt-oss-20b:free"])

if st.sidebar.button("开始提取") and pdf_file and api_key:
    # 临时文件保存 PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.read())
        pdf_path = tmp_pdf.name

    st.info("正在读取 PDF 内容…")
    pdf_reader = PdfReader(pdf_path)
    pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)

    st.info("正在调用 LLM 模型提取方法结构…")
    start_time = time.time()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    from utils.prompt import system_prompt_method_v3  # 需要你本地有此模块

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_method_v3},
            {"role": "user", "content": f"Paper content: {pdf_text}"}
        ]
    )
    result = completion.choices[0].message.content
    result = result.strip("```json\n").strip("```")
    end_time = time.time()

    st.success(f"✅ 图谱提取完成！用时 {end_time - start_time:.2f} 秒")

    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    json_path = f"method_{timestamp}.json"
    try:
        json_data = json.loads(result)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        st.download_button("📥 下载 JSON 结果", open(json_path, "rb"), file_name=json_path)
    except json.JSONDecodeError:
        st.error("❌ JSON 解析失败，显示原始输出：")
        st.text(result)
        st.stop()

    # ========== 可视化 ==========
    st.info("正在生成图谱可视化…")
    G = nx.DiGraph()

    for node in json_data["nodes"]:
        G.add_node(node["id"], label=node["canonical_name"], confidence=node["confidence_ie"])

    for edge in json_data["edges"]:
        G.add_edge(edge["source_id"], edge["target_id"], relation=edge["relation"])

    net = Network(height="750px", width="100%", bgcolor="#ffffff", directed=True)
    net.set_options("""
    {
      "nodes": {"font": {"size": 20, "color": "#000000"}},
      "edges": {"font": {"size": 15, "color": "#000000"}},
      "physics": {"barnesHut": {"gravity": -8000, "centralGravity": 0.1, "springLength": 200}}
    }
    """)

    unique_names = set(node["canonical_name"] for node in json_data["nodes"])
    color_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FDCB6E", "#E17055", "#A29BFE",
                     "#636E72", "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43",
                     "#10AC84", "#EE5A24"]
    color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(unique_names)}
    color_map["Default"] = "#636E72"

    for node_id, data in G.nodes(data=True):
        name = data["label"]
        size = 5 + 35 * data["confidence"]
        color = next((v for k, v in color_map.items() if k in name), color_map["Default"])
        net.add_node(node_id, label=name,
                     title=f"<b>{name}</b><br>Confidence: {data['confidence']:.2f}",
                     size=size, color=color)

    for u, v, data in G.edges(data=True):
        rel = data["relation"]
        net.add_edge(u, v, title=rel, label=rel, color="#2d3436")

    html_path = f"method_{timestamp}.html"
    net.write_html(html_path)
    st.success("✅ 图谱生成成功！")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)

else:
    st.warning("请在左侧上传 PDF 并输入 API Key 后点击“开始提取”")

