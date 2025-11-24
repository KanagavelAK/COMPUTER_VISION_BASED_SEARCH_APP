import streamlit as st
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple

# If your project has src package with inference and utils, keep these imports.
# Otherwise adapt or replace the inference calls with your own implementation.
try:
    from src.inference import YOLOv11Inference
    from src.utils import save_metadata, load_metadata, get_unique_classes_counts
except Exception:
    # fallbacks if running outside package during development
    YOLOv11Inference = None
    def save_metadata(md, image_dir):
        out = Path(image_dir) / "metadata.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(md, f, indent=2)
        return out

    def load_metadata(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_unique_classes_counts(metadata):
        classes = set()
        counts = {}
        for it in metadata:
            for k, v in it.get("class_counts", {}).items():
                classes.add(k)
                counts.setdefault(k, set()).add(v)
        # convert sets to sorted lists
        counts = {k: sorted(list(v)) for k, v in counts.items()}
        return sorted(list(classes)), counts


st.set_page_config(page_title="YOLOv11 Search App", layout="wide")

# ------------------- Helpers -------------------

def sanitize_path(s: str) -> str:
    if not s:
        return ""
    return s.strip().strip('"').strip("'")


def img_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buff = io.BytesIO()
    img.save(buff, format=fmt)
    return buff.getvalue()


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        return img.convert("RGB")
    return img


def make_thumbnail_and_scale(img: Image.Image, max_size: Tuple[int, int] = (900, 675)) -> Tuple[Image.Image, float, float]:
    orig_w, orig_h = img.size
    thumb = img.copy()
    thumb.thumbnail(max_size)
    tw, th = thumb.size
    sx = tw / orig_w
    sy = th / orig_h
    return thumb, sx, sy


# ------------------- Session state init -------------------

def init_session_state():
    defaults = {
        "metadata": None,
        "unique_classes": [],
        "count_options": {},
        "search_results": [],
        "search_params": {
            "search_mode": "Any of selected classes (OR)",
            "selected_classes": [],
            "thresholds": {}
        },
        "show_boxes": True,
        "grid_columns": 3,
        "highlight_matches": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ------------------- CSS (keep minimal and resilient) -------------------
st.markdown(
    """
<style>
.image-card{ border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:16px; background:#fff}
.image-meta{ padding:8px; background:rgba(0,0,0,0.7); color:#fff; font-size:12px}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------- UI -------------------
st.title("Computer Vision Powered Search Application")

option = st.radio("Choose an option:", ("Process new images", "Load existing metadata"), horizontal=True)

if option == "Process new images":
    with st.expander("Process new images", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            image_dir_input = st.text_input("Image directory path:", placeholder=r"C:\path\to\images")
            image_dir = sanitize_path(image_dir_input)
        with col2:
            model_path_input = st.text_input("Model weights path:", value="yolo11m.pt")
            model_path = sanitize_path(model_path_input)

        # provide a quick validation to help users
        if image_dir and not Path(image_dir).exists():
            st.warning("Image directory not found. Check path and remove extra quotes if any.")
        if model_path and not Path(model_path).exists():
            st.info("Model file not found at given path. Make sure the model is accessible from the working directory or provide full path.")

        if st.button("Start Inference"):
            if not image_dir:
                st.warning("Please enter an image directory path")
            else:
                if YOLOv11Inference is None:
                    st.error("Inference backend not available. Make sure src.inference.YOLOv11Inference is importable.")
                else:
                    try:
                        with st.spinner("Running object detection on images..."):
                            inferencer = YOLOv11Inference(model_path)
                            metadata = inferencer.process_directory(image_dir)
                            metadata_path = save_metadata(metadata, image_dir)
                            st.success(f"Processed {len(metadata)} images. Metadata saved to: {metadata_path}")

                            st.session_state.metadata = metadata
                            st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                    except Exception as e:
                        st.exception(e)

else:
    with st.expander("Load Existing Metadata", expanded=True):
        metadata_path_input = st.text_input("Metadata file path:", placeholder="path/to/metadata.json")
        metadata_path = sanitize_path(metadata_path_input)

        if st.button("Load Metadata"):
            if not metadata_path:
                st.warning("Please enter a metadata file path")
            else:
                try:
                    with st.spinner("Loading metadata..."):
                        metadata = load_metadata(metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        st.success(f"Loaded metadata for {len(metadata)} images")
                except Exception as e:
                    st.exception(e)

# ------------------- Search UI -------------------
if st.session_state.metadata:
    st.header("🔍 Search Engine")
    with st.container():
        st.session_state.search_params["search_mode"] = st.radio(
            "Search mode:", ("Any of selected classes (OR)", "All selected classes (AND)"), horizontal=True
        )

        selected = st.multiselect("Classes to search for:", options=st.session_state.unique_classes, key="_class_select")
        st.session_state.search_params["selected_classes"] = selected

        # cleanup thresholds for removed classes
        st.session_state.search_params["thresholds"] = {
            k: v
            for k, v in st.session_state.search_params.get("thresholds", {}).items()
            if k in selected
        }

        if selected:
            st.subheader("Count Thresholds (optional)")
            cols = st.columns(len(selected))
            for i, cls in enumerate(selected):
                with cols[i]:
                    options = ["None"] + st.session_state.count_options.get(cls, [])
                    chosen = st.selectbox(f"Max count for {cls}", options=options, key=f"thr_{cls}")
                    st.session_state.search_params["thresholds"][cls] = chosen

        if st.button("Search Images") and st.session_state.search_params["selected_classes"]:
            results = []
            search_params = st.session_state.search_params
            for item in st.session_state.metadata:
                class_matches = {}
                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item.get("detections", []) if d.get("class") == cls]
                    class_count = len(class_detections)
                    threshold = search_params["thresholds"].get(cls, "None")
                    if threshold == "None":
                        class_matches[cls] = class_count >= 1
                    else:
                        try:
                            class_matches[cls] = (class_count >= 1 and class_count <= int(threshold))
                        except Exception:
                            class_matches[cls] = (class_count >= 1)

                if search_params["search_mode"] == "Any of selected classes (OR)":
                    matches = any(class_matches.values())
                else:
                    matches = all(class_matches.values())

                if matches:
                    results.append(item)

            st.session_state.search_results = results

# ------------------- Display Results -------------------
if st.session_state.search_results:
    results = st.session_state.search_results
    search_params = st.session_state.search_params

    st.subheader(f"📷 Results: {len(results)} matching images")

    with st.expander("Display Options", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.session_state.show_boxes = st.checkbox("Show bounding boxes", value=st.session_state.show_boxes)
        with cols[1]:
            st.session_state.grid_columns = st.slider("Grid columns", min_value=2, max_value=6, value=st.session_state.grid_columns)
        with cols[2]:
            st.session_state.highlight_matches = st.checkbox("Highlight matching classes", value=st.session_state.highlight_matches)

    grid_cols = st.columns(st.session_state.grid_columns)
    col_index = 0

    for result in results:
        with grid_cols[col_index]:
            try:
                img_path = Path(result["image_path"])
                if not img_path.exists():
                    st.error(f"Image not found: {img_path}")
                else:
                    img = Image.open(img_path)
                    img = ensure_rgb(img)

                    display_img, sx, sy = make_thumbnail_and_scale(img)
                    draw = ImageDraw.Draw(display_img)

                    # load font (fallback to default)
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except Exception:
                        font = ImageFont.load_default()

                    for det in result.get("detections", []):
                        cls = det.get("class")
                        bbox = det.get("bbox", [0, 0, 0, 0])
                        # support bboxes as dict or list
                        if isinstance(bbox, dict):
                            bbox = [bbox.get(k, 0) for k in ("x1", "y1", "x2", "y2")]

                        # scale bbox and coerce to ints
                        try:
                            bx = int(round(bbox[0] * sx))
                            by = int(round(bbox[1] * sy))
                            bx2 = int(round(bbox[2] * sx))
                            by2 = int(round(bbox[3] * sy))
                            bbox_disp = [bx, by, bx2, by2]
                        except Exception:
                            bbox_disp = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                        if cls in search_params["selected_classes"]:
                            color = "#30C938"
                            thickness = 3
                        elif not st.session_state.highlight_matches:
                            color = "#666666"
                            thickness = 1
                        else:
                            continue

                        if st.session_state.show_boxes:
                            draw.rectangle(bbox_disp, outline=color, width=thickness)

                            if cls in search_params["selected_classes"] or not st.session_state.highlight_matches:
                                label = f"{cls} {det.get('confidence', 0):.2f}"
                                try:
                                    text_bbox = draw.textbbox((0, 0), label, font=font)
                                    text_w = text_bbox[2] - text_bbox[0]
                                    text_h = text_bbox[3] - text_bbox[1]
                                except Exception:
                                    text_w, text_h = draw.textsize(label, font=font)

                                draw.rectangle([bbox_disp[0], bbox_disp[1], bbox_disp[0] + text_w + 8, bbox_disp[1] + text_h + 4], fill=color)
                                draw.text((bbox_disp[0] + 4, bbox_disp[1] + 2), label, fill="white", font=font)

                    # prepare meta overlay
                    meta_items = [f"{k}: {v}" for k, v in result.get("class_counts", {}).items() if k in search_params["selected_classes"]]
                    caption = ", ".join(meta_items) if meta_items else "No matches"

                    # render card: image + caption
                    st.markdown("<div class=\"image-card\">", unsafe_allow_html=True)
                    st.image(display_img, use_column_width=True, caption=Path(result["image_path"]).name)
                    st.markdown(f"<div class=\"image-meta\">{caption}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error displaying {result.get('image_path')}: {e}")

        col_index = (col_index + 1) % st.session_state.grid_columns

    with st.expander("Export Options"):
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name="search_results.json",
            mime="application/json",
        )

# ------------------- Footer / tips -------------------
st.caption("Tips: avoid surrounding path strings with quotes. Use full absolute paths when in doubt.")
