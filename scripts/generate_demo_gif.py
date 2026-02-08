#!/usr/bin/env python3
"""Generate terminal-style demo GIFs for README."""

from PIL import Image, ImageDraw, ImageFont
import os

# --- Config ---
WIDTH = 720
HEIGHT = 420
BG_COLOR = (30, 30, 46)         # Dark background
TEXT_COLOR = (205, 214, 244)     # Light text
PROMPT_COLOR = (166, 227, 161)   # Green prompt
CMD_COLOR = (137, 180, 250)      # Blue command
OUTPUT_COLOR = (186, 194, 222)   # Muted output
COMMENT_COLOR = (108, 112, 134)  # Gray comment
SUCCESS_COLOR = (166, 227, 161)  # Green success
TITLE_BAR_COLOR = (49, 50, 68)   # Title bar
TITLE_TEXT_COLOR = (186, 194, 222)
CURSOR_COLOR = (245, 224, 220)   # Cursor

LINE_HEIGHT = 20
LEFT_MARGIN = 16
TOP_MARGIN = 40  # Below title bar
TITLE_BAR_HEIGHT = 30

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def get_font(size=14):
    """Get a font that supports Japanese characters."""
    jp_paths = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in jp_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_title_bar(draw, title, font):
    """Draw macOS-style title bar."""
    draw.rectangle([0, 0, WIDTH, TITLE_BAR_HEIGHT], fill=TITLE_BAR_COLOR)
    # Traffic lights
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([12 + i * 22, 8, 26 + i * 22, 22], fill=color)
    # Title
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 7), title, fill=TITLE_TEXT_COLOR, font=font)


def create_frame(lines, font, title="Terminal", cursor_pos=None):
    """Create a single frame with given lines."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_title_bar(draw, title, font)

    y = TOP_MARGIN + 8
    for line in lines:
        x = LEFT_MARGIN
        if isinstance(line, list):
            # List of (text, color) tuples
            for text, color in line:
                draw.text((x, y), text, fill=color, font=font)
                bbox = draw.textbbox((0, 0), text, font=font)
                x += bbox[2] - bbox[0]
        else:
            draw.text((x, y), line[0], fill=line[1] if len(line) > 1 else TEXT_COLOR, font=font)
        y += LINE_HEIGHT

    # Draw cursor if specified
    if cursor_pos is not None:
        cx, cy = cursor_pos
        draw.rectangle([cx, cy, cx + 8, cy + LINE_HEIGHT - 2], fill=CURSOR_COLOR)

    return img


def typing_frames(base_lines, prompt_parts, cmd_text, font, title, ms_per_char=80):
    """Generate frames that show a command being typed character by character."""
    frames = []
    durations = []
    for i in range(len(cmd_text) + 1):
        partial = cmd_text[:i]
        current_line = list(prompt_parts) + [(partial, CMD_COLOR)]
        lines = base_lines + [current_line]
        frames.append(create_frame(lines, font, title))
        durations.append(ms_per_char)
    # Pause at the end
    frames.append(create_frame(base_lines + [list(prompt_parts) + [(cmd_text, CMD_COLOR)]], font, title))
    durations.append(400)
    return frames, durations


def generate_setup_gif():
    """Generate the setup & launch demo GIF."""
    font = get_font(14)
    title_font = get_font(13)
    frames = []
    durations = []

    prompt = [("$ ", PROMPT_COLOR)]

    # --- Scene 1: Clone ---
    scene_lines = []
    comment = [("# 1. リポジトリをクローン", COMMENT_COLOR)]
    scene_lines.append(comment)

    # Show comment for a moment
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(1200)

    # Type clone command
    cmd = "git clone https://github.com/shigenoburyuto/nazono-kokka.git"
    f, d = typing_frames(scene_lines, prompt, cmd, font, "Terminal — セットアップ", 40)
    frames.extend(f)
    durations.extend(d)

    # Output
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd, CMD_COLOR)])
    scene_lines.append([("Cloning into 'nazono-kokka'...", OUTPUT_COLOR)])
    scene_lines.append([("done.", SUCCESS_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(800)

    # cd
    scene_lines.append([])
    cmd2 = "cd nazono-kokka"
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd2, CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(600)

    # --- Scene 2: venv ---
    scene_lines.append([])
    scene_lines.append([("# 2. 仮想環境を作成", COMMENT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(800)

    cmd3 = "python3 -m venv .venv && source .venv/bin/activate"
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd3, CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(600)

    cmd4 = "pip install -r requirements.txt"
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd4, CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(400)

    scene_lines.append([("Successfully installed 42 packages", SUCCESS_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(1000)

    # Hold final frame
    frames.append(create_frame(scene_lines, font, "Terminal — セットアップ"))
    durations.append(2000)

    # Save
    path = os.path.join(ASSETS_DIR, "demo-setup.gif")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Created: {path}")


def generate_launch_gif():
    """Generate the launch & usage demo GIF."""
    font = get_font(14)
    frames = []
    durations = []

    prompt = [("$ ", PROMPT_COLOR)]
    scene_lines = []

    # --- Ollama ---
    scene_lines.append([("# 1. Ollama を起動してモデルを取得", COMMENT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(1000)

    cmd1 = "ollama serve &"
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd1, CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(500)

    scene_lines.append([("Ollama is running on http://localhost:11434", OUTPUT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(600)

    scene_lines.append([])
    cmd2 = "ollama pull schroneko/llama-3.1-swallow-8b-instruct-v0.1"
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd2, CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(500)

    scene_lines.append([("pulling manifest... done", OUTPUT_COLOR)])
    scene_lines.append([("success", SUCCESS_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(800)

    # --- Ingest ---
    scene_lines.append([])
    scene_lines.append([("# 2. データをベクトル DB に取り込み", COMMENT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(800)

    cmd3 = "python -m rag_system.ingest"
    scene_lines.append([("$ ", PROMPT_COLOR), (cmd3, CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(500)

    scene_lines.append([("Loading legal framework documents...", OUTPUT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(400)

    scene_lines.append([("Loading precedents (1,206 cases)...", OUTPUT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(400)

    scene_lines.append([("Ingested 1,354 chunks into ChromaDB", SUCCESS_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(1000)

    # --- Launch ---
    scene_lines.append([])
    scene_lines.append([("# 3. Web UI を起動", COMMENT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(800)

    cmd4 = "streamlit run app.py"
    f, d = typing_frames(scene_lines, prompt, cmd4, font, "Terminal — 起動", 60)
    frames.extend(f)
    durations.extend(d)

    scene_lines.append([("$ ", PROMPT_COLOR), (cmd4, CMD_COLOR)])
    scene_lines.append([])
    scene_lines.append([("  You can now view your Streamlit app in your browser.", OUTPUT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(500)

    scene_lines.append([("  Local URL: ", OUTPUT_COLOR), ("http://localhost:8501", CMD_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(2500)

    # Hold
    frames.append(create_frame(scene_lines, font, "Terminal — 起動"))
    durations.append(2000)

    path = os.path.join(ASSETS_DIR, "demo-launch.gif")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Created: {path}")


def generate_cli_gif():
    """Generate the CLI usage demo GIF."""
    font = get_font(14)
    frames = []
    durations = []

    prompt = [("(.venv) $ ", PROMPT_COLOR)]
    scene_lines = []

    # --- Query ---
    scene_lines.append([("# CLI で単発クエリ", COMMENT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — CLI"))
    durations.append(1000)

    cmd = 'python rag_system/main.py --query "窃盗罪の量刑基準を示せ"'
    f, d = typing_frames(scene_lines, prompt, cmd, font, "Terminal — CLI", 45)
    frames.extend(f)
    durations.extend(d)

    scene_lines.append([("(.venv) $ ", PROMPT_COLOR), (cmd, CMD_COLOR)])
    scene_lines.append([])
    scene_lines.append([("=== 司法判断 ===", (249, 226, 175))])
    frames.append(create_frame(scene_lines, font, "Terminal — CLI"))
    durations.append(800)

    scene_lines.append([])
    scene_lines.append([("【適用法令】", (249, 226, 175))])
    scene_lines.append([("  刑法第204条（窃盗罪）", TEXT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — CLI"))
    durations.append(600)

    scene_lines.append([])
    scene_lines.append([("【量刑基準】", (249, 226, 175))])
    scene_lines.append([("  基本量刑: 1年以上10年以下の懲役", TEXT_COLOR)])
    scene_lines.append([("  または50万トーン以下の罰金", TEXT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — CLI"))
    durations.append(800)

    scene_lines.append([])
    scene_lines.append([("【参照判例】", (249, 226, 175))])
    scene_lines.append([("  CRIM-2019-0042 (有罪・懲役2年)", OUTPUT_COLOR)])
    scene_lines.append([("  CRIM-2021-0187 (有罪・懲役3年6月)", OUTPUT_COLOR)])
    frames.append(create_frame(scene_lines, font, "Terminal — CLI"))
    durations.append(1500)

    # Hold
    frames.append(create_frame(scene_lines, font, "Terminal — CLI"))
    durations.append(3000)

    path = os.path.join(ASSETS_DIR, "demo-cli.gif")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Created: {path}")


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    generate_setup_gif()
    generate_launch_gif()
    generate_cli_gif()
    print("\nAll demo GIFs generated!")
