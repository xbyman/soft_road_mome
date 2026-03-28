"""
Road-MoME 推理与可视化引擎 (v11.0 四格热力图答辩展示版)

在 v10.6 基础上修改：
- draw_inference_result → draw_4panel_result
  保留完全相同的单图渲染逻辑（fillPoly/polylines/addWeighted 0.1 alpha），
  改为用 INFERNO colormap 渲染四张热力图并拼接：
    左上：预测概率热力图  右上：Phys 专家权重热力图
    左下：Geom 专家权重   右下：Tex 专家权重
- main() 逻辑、路径、索引、模型加载全部保持不变
"""

import os
import sys
import json
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.mome_model import MoMEEngine


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generate_rois(
    patch_corners_uv, valid_mask, orig_w, orig_h, target_w=1008, target_h=560
):
    rois = []
    scale_x, scale_y = target_w / orig_w, target_h / orig_h
    for i in range(63):
        if valid_mask[i] == 0:
            rois.append([0.0, 0.0, 1.0, 1.0])
        else:
            corners = patch_corners_uv[i]
            x_min = max(0, np.min(corners[:, 0]) * scale_x)
            x_max = min(target_w, np.max(corners[:, 0]) * scale_x)
            y_min = max(0, np.min(corners[:, 1]) * scale_y)
            y_max = min(target_h, np.max(corners[:, 1]) * scale_y)
            if x_max <= x_min or y_max <= y_min:
                rois.append([0.0, 0.0, 1.0, 1.0])
            else:
                rois.append([float(x_min), float(y_min), float(x_max), float(y_max)])

    rois_tensor = torch.tensor(rois, dtype=torch.float32)
    batch_idx = torch.zeros((63, 1), dtype=torch.float32)
    return torch.cat([batch_idx, rois_tensor], dim=1)


def val_to_inferno_bgr(v):
    """将 [0,1] 标量映射到 INFERNO colormap，返回 BGR tuple。"""
    stops = np.array(
        [
            [0, 0, 4],
            [40, 11, 84],
            [101, 21, 110],
            [159, 42, 99],
            [212, 72, 66],
            [245, 125, 21],
            [252, 193, 33],
            [252, 255, 164],
        ],
        dtype=np.float32,
    )
    v = float(np.clip(v, 0.0, 1.0))
    idx = v * (len(stops) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(stops) - 1)
    rgb = stops[lo] * (1 - (idx - lo)) + stops[hi] * (idx - lo)
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # BGR


def render_panel(img_bgr, patch_corners_uv, valid_mask, values, title):
    """
    单张热力图面板渲染。
    逻辑与原始 draw_inference_result 完全一致：
      fillPoly → overlay，polylines → img，addWeighted(overlay, 0.1, img, 0.9)
    坐标直接用 patch_corners_uv（原始图像坐标），不做任何缩放。
    """
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()
    overlay = img_bgr.copy()

    for i in range(63):
        if valid_mask[i] == 0:
            continue
        corners = patch_corners_uv[i].astype(np.int32)
        color = val_to_inferno_bgr(values[i])
        cv2.fillPoly(overlay, [corners], color)
        cv2.polylines(
            img,
            [corners],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    # 面板顶部信息栏（与原始脚本样式一致）
    header = np.zeros((60, w, 3), dtype=np.uint8)
    header[:] = (45, 45, 45)
    cv2.putText(
        header,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([header, img])


def draw_4panel_result(
    img_bgr, patch_corners_uv, valid_mask, probs, weights, frame_id, out_dir
):
    """
    生成 2×2 四格热力图，带总信息栏，保存为 JPG。
    weights shape: (1, 63, 3)，与原始脚本一致。
    """
    h, w = img_bgr.shape[:2]

    valid_idx = valid_mask > 0.5
    n_valid = int(valid_idx.sum())
    if n_valid > 0:
        w_p = weights[0, valid_idx, 0].mean()
        w_g = weights[0, valid_idx, 1].mean()
        w_t = weights[0, valid_idx, 2].mean()
        avg_prob = probs[valid_idx].mean()
    else:
        w_p = w_g = w_t = avg_prob = 0.0

    panel_pred = render_panel(
        img_bgr, patch_corners_uv, valid_mask, probs, f"Prediction  avg={avg_prob:.2f}"
    )
    panel_phys = render_panel(
        img_bgr,
        patch_corners_uv,
        valid_mask,
        weights[0, :, 0],
        f"Phys expert  w={w_p:.2f}",
    )
    panel_geom = render_panel(
        img_bgr,
        patch_corners_uv,
        valid_mask,
        weights[0, :, 1],
        f"Geom expert  w={w_g:.2f}",
    )
    panel_tex = render_panel(
        img_bgr,
        patch_corners_uv,
        valid_mask,
        weights[0, :, 2],
        f"Tex expert   w={w_t:.2f}",
    )

    # 2×2 拼接
    row_top = np.hstack([panel_pred, panel_phys])
    row_bottom = np.hstack([panel_geom, panel_tex])
    grid = np.vstack([row_top, row_bottom])

    # ── colorbar 色条（横跨全宽，标注 INFERNO 颜色与值的对应关系）──
    total_w = grid.shape[1]
    cb_h = 36  # 色条本体高度
    label_h = 24  # 刻度文字高度
    bar_strip = np.zeros((cb_h, total_w, 3), dtype=np.uint8)
    for x in range(total_w):
        bar_strip[:, x] = val_to_inferno_bgr(x / max(total_w - 1, 1))

    # 刻度文字行（深灰背景）
    label_strip = np.zeros((label_h, total_w, 3), dtype=np.uint8)
    label_strip[:] = (30, 30, 30)
    for val, label in [(0.0, "0.0  low"), (0.5, "0.5"), (1.0, "high  1.0")]:
        x = int(val * (total_w - 1))
        anchor = "left" if val == 0.0 else ("right" if val == 1.0 else "center")
        if anchor == "center":
            text_x = max(0, x - 16)
        elif anchor == "right":
            text_x = max(0, x - 88)
        else:
            text_x = x + 4
        cv2.putText(
            label_strip,
            label,
            (text_x, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    colorbar = np.vstack([bar_strip, label_strip])

    # ── 总信息栏 ──
    banner = np.zeros((60, total_w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    info = (
        f"Frame: {frame_id}    Valid: {n_valid}/63    "
        f"Phys: {w_p:.2f}  Geom: {w_g:.2f}  Tex: {w_t:.2f}    "
        f"Avg pred prob: {avg_prob:.3f}"
    )
    cv2.putText(
        banner,
        info,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    # 顺序：总信息栏 → colorbar → 四格图
    final_canvas = np.vstack([banner, colorbar, grid])

    # 等比例安全缩小（与原始脚本逻辑一致，2×2 拼图放宽到 4K）
    MAX_OUT_WIDTH = 3840
    fh, fw = final_canvas.shape[:2]
    if fw > MAX_OUT_WIDTH:
        ratio = MAX_OUT_WIDTH / fw
        final_canvas = cv2.resize(
            final_canvas, (MAX_OUT_WIDTH, int(fh * ratio)), interpolation=cv2.INTER_AREA
        )

    save_path = Path(out_dir) / f"vis_{frame_id}.jpg"
    cv2.imwrite(str(save_path), final_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # 【核心修复区】：绝对路径锚定机制 (与训练脚本保持一致)
    # ====================================================
    def get_abs_path(p_str):
        p_str = str(p_str)
        if p_str.startswith("./"):
            return project_root / p_str[2:]
        if Path(p_str).is_absolute():
            return Path(p_str)
        return project_root / p_str

    npz_dir = get_abs_path(cfg["paths"]["output_dir"])
    img_dir = get_abs_path(cfg["paths"]["raw_img_dir"])
    vis_out_dir = get_abs_path(
        cfg["paths"].get("vis_output_dir", "./data/inference_vis_v9.8")
    )
    json_gt_path = get_abs_path(
        cfg["paths"].get("manual_label_path", "./data/merged_visual_gt.json")
    )

    vis_out_dir.mkdir(parents=True, exist_ok=True)

    # 挂载人工真值库
    manual_gt_dict = {}
    if json_gt_path.exists():
        with open(json_gt_path, "r", encoding="utf-8") as f:
            manual_gt_dict = json.load(f)
        print(
            f"✅ 成功加载人工真值标签库: {json_gt_path.name} (共 {len(manual_gt_dict)} 帧)"
        )
    else:
        print(f"⚠️ 警告: 未找到人工标签库 {json_gt_path}，将全部使用 3D 伪标签！")
    # ====================================================

    print("⏳ 正在初始化 MoME E2E 架构与加载权重...")
    dim_phys = cfg["features"]["phys"]["input_dim"]
    dim_3d = cfg["features"]["3d"]["input_dim"]
    model = MoMEEngine(dim_f3_stats=dim_phys, dim_f3_mae=dim_3d).to(device)

    weight_path = get_abs_path(cfg["paths"]["weights"]["mome_model"])
    if not weight_path.exists():
        print(f"❌ 找不到最优权重文件: {weight_path}")
        return

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    # ==========================================================
    # 【核心重构：精准索引制导加载机制】
    # ==========================================================
    img_map = {}

    # 扩大索引文件的搜索半径，兼容根目录与 data 目录
    possible_index_paths = [
        project_root / "data" / "dataset_index.yaml",
        project_root / "data" / "dataset_index.json",
        project_root / "dataset_index.yaml",
        project_root / "dataset_index.json",
    ]

    index_loaded = False
    for path in possible_index_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".yaml":
                    index_data = yaml.safe_load(f)
                else:
                    index_data = json.load(f)
                # 强制转 str 防止 YAML 解析错误
                img_map = {str(item["id"]): Path(item["img"]) for item in index_data}
            print(f"📖 成功挂载标准数据集索引: {path} (共映射 {len(img_map)} 帧)")
            index_loaded = True
            break

    if not index_loaded:
        print("⚠️ 未检测到 dataset_index.yaml/json，回退到 rglob 盲搜模式...")
        all_imgs = list(img_dir.rglob("*.jpg"))
        img_map = {p.stem: p for p in all_imgs}
    # ==========================================================

    npz_files = sorted(list(npz_dir.glob("pkg_*.npz")))
    valid_files = [f for f in npz_files if f.stem.replace("pkg_", "") in img_map]

    limit = cfg.get("inference", {}).get("batch_limit", 20)
    if len(valid_files) > limit:
        np.random.seed(42)
        test_files = np.random.choice(valid_files, limit, replace=False)
    else:
        test_files = valid_files

    ablation_cfg = cfg.get("training", {}).get("ablation", {})
    if any(ablation_cfg.values()):
        print(f"⚠️  消融模式: {ablation_cfg}")

    print(f"🚀 启动【精准索引版】推理可视化 | 测试帧数: {len(test_files)}")

    with torch.no_grad():
        for npz_path in tqdm(test_files, desc="Inference & Render"):
            frame_id = npz_path.stem.replace("pkg_", "")

            with np.load(npz_path, allow_pickle=True) as data:
                f3_stats = (
                    torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(device)
                )
                f3_mae = (
                    torch.from_numpy(data["deep_512d"]).float().unsqueeze(0).to(device)
                )
                q_2d = (
                    torch.from_numpy(
                        data.get("quality_2d", np.ones((63, 1), dtype=np.float32))
                    )
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                patch_corners_uv = data["patch_corners_uv"]
                meta = data["meta"]

            pseudo_label = meta[:, 0]
            valid_mask = meta[:, 2]

            img_path = img_map[frame_id]
            try:
                # 兼容 Windows/Linux 中文或复杂路径的绝对安全读取法
                img_bgr = cv2.imdecode(
                    np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
                )
            except Exception:
                img_bgr = None

            if img_bgr is None:
                print(f"\n⚠️ 无法读取图像: {img_path}")
                continue

            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1008, 560))
            img_tensor = img_transform(img_resized).unsqueeze(0).to(device)

            rois_tensor = generate_rois(
                patch_corners_uv, valid_mask, orig_w, orig_h
            ).to(device)
            valid_mask_tensor = (
                torch.from_numpy(valid_mask).float().unsqueeze(0).to(device)
            )

            final_logit, internals = model(
                img_tensor,
                rois_tensor,
                f3_stats,
                f3_mae,
                q_2d,
                valid_mask_tensor,
                ablation_cfg=ablation_cfg,
            )

            probs = torch.sigmoid(final_logit).squeeze(0).cpu().numpy()  # (63,)
            weights = internals["weights"].cpu().numpy()  # (1, 63, 3)

            draw_4panel_result(
                img_bgr=img_bgr,
                patch_corners_uv=patch_corners_uv,
                valid_mask=valid_mask,
                probs=probs,
                weights=weights,
                frame_id=frame_id,
                out_dir=vis_out_dir,
            )

    print(f"\n✨ 渲染完成! 结果已保存至: {vis_out_dir}")


if __name__ == "__main__":
    main()
