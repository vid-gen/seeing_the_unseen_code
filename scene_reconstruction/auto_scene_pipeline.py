#!/usr/bin/env python3
"""
SAM 3D Objects 多物体场景重建 Pipeline

使用 SAM3 文本提示分割 + SAM 3D Objects 从单张图像生成 3D Gaussian Splat 场景

工作流:
1. SAM3 文本提示检测并分割前景物体
2. SAM 3D Objects 对每个物体生成 3D Gaussian Splat（含位姿）
3. make_scene() 合并所有物体到统一场景坐标系
4. 保存输出：场景 PLY、渲染 GIF、中间结果
"""

import os
import sys
import argparse
import json
import base64
import requests
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import cv2

# ============== 环境设置（必须在导入模型之前）==============
# HuggingFace cache: use the standard HF_HOME environment variable (not hard-coded here).
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
os.environ['LIDRA_SKIP_INIT'] = 'true'
# SAM 3D Objects 的 inference.py 需要 CONDA_PREFIX
if 'CONDA_PREFIX' not in os.environ:
    os.environ['CONDA_PREFIX'] = '/usr/local/cuda'

# ============== 补丁: 添加缺失函数到 utils3d.numpy ==============
def _patch_utils3d():
    """添加缺失函数到 utils3d.numpy（SAM 3D Objects 需要）"""
    try:
        import utils3d.numpy as u3d_np
    except ImportError:
        return  # utils3d 未安装，跳过

    # 检查是否已有 depth_edge
    if hasattr(u3d_np, 'depth_edge'):
        return

    def depth_edge(depth, rtol=0.03, mask=None):
        """检测深度图中的边缘（深度不连续区域）"""
        if mask is None:
            mask = np.ones_like(depth, dtype=bool)
        gy, gx = np.gradient(depth)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        depth_nonzero = np.where(depth > 0, depth, 1.0)
        relative_gradient = gradient_mag / depth_nonzero
        edge_mask = (relative_gradient > rtol) & mask
        return edge_mask

    def normals_edge(normals, atol=0.1, tol=None, mask=None):
        """检测法线不连续的边缘"""
        if tol is not None:
            atol = tol
        if mask is None:
            mask = np.ones(normals.shape[:2], dtype=bool)
        gy, gx = np.gradient(normals, axis=(0, 1))
        gradient_mag = np.sqrt(np.sum(gx**2 + gy**2, axis=-1))
        edge_mask = (gradient_mag > atol) & mask
        return edge_mask

    def points_to_normals(points, mask=None):
        """从点云计算法线"""
        if mask is None:
            mask = np.ones(points.shape[:2], dtype=bool)
        dy = np.zeros_like(points)
        dx = np.zeros_like(points)
        dy[:-1, :, :] = points[1:, :, :] - points[:-1, :, :]
        dx[:, :-1, :] = points[:, 1:, :] - points[:, :-1, :]
        normals = np.cross(dx, dy)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        norm = np.where(norm > 1e-8, norm, 1.0)
        normals = normals / norm
        normals = np.where(normals[..., 2:3] > 0, -normals, normals)
        normals_mask = mask & (np.linalg.norm(normals, axis=-1) > 0.5)
        return normals, normals_mask

    def image_uv(height, width):
        """生成图像 UV 坐标网格"""
        u = np.linspace(0, 1, width)
        v = np.linspace(0, 1, height)
        uu, vv = np.meshgrid(u, v)
        uv = np.stack([uu, vv], axis=-1)
        return uv

    def image_mesh(height, width, mask=None):
        """从图像生成三角面片"""
        if mask is None:
            mask = np.ones((height, width), dtype=bool)
        vertex_idx = np.arange(height * width).reshape(height, width)
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                if mask[i, j] and mask[i+1, j] and mask[i, j+1] and mask[i+1, j+1]:
                    v0 = vertex_idx[i, j]
                    v1 = vertex_idx[i+1, j]
                    v2 = vertex_idx[i, j+1]
                    v3 = vertex_idx[i+1, j+1]
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        if len(faces) == 0:
            return np.zeros((0, 3), dtype=np.int32), vertex_idx
        return np.array(faces, dtype=np.int32), vertex_idx

    # 添加到 utils3d.numpy
    u3d_np.depth_edge = depth_edge
    u3d_np.normals_edge = normals_edge
    u3d_np.points_to_normals = points_to_normals
    u3d_np.image_uv = image_uv
    u3d_np.image_mesh = image_mesh

# 应用补丁
_patch_utils3d()

# 添加 notebook 路径以使用 SAM 3D Objects 的 inference 模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Root of the SAM-3D-Objects checkout (contains notebook/ and checkpoints/hf/).
# Set the SAM3D_ROOT env variable; otherwise this folder is assumed to sit inside the checkout.
SAM3D_ROOT = os.environ.get('SAM3D_ROOT', os.path.dirname(SCRIPT_DIR))
NOTEBOOK_DIR = os.path.join(SAM3D_ROOT, 'notebook')

if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)
if SAM3D_ROOT not in sys.path:
    sys.path.insert(0, SAM3D_ROOT)


class AutoScenePipeline:
    """
    自动多物体场景重建 Pipeline

    使用 SAM3 文本提示分割检测并分割物体，再通过 SAM 3D Objects 进行 3D 重建。
    """

    def __init__(
        self,
        device: str = "cuda",
        sam3d_config: str = None,
        compile_model: bool = False
    ):
        """
        初始化 Pipeline

        Args:
            device: 计算设备
            sam3d_config: SAM 3D Objects 配置路径
            compile_model: 是否编译 SAM 3D 模型（慢启动，快推理）
        """
        self.device = device
        self.compile_model = compile_model

        # 延迟加载的模型
        self.sam3d_inference = None

        # MoGe 推理缓存（避免重复调用）
        self._moge_cache = None

        # SAM3 相关（文本提示分割）
        self._sam3_model = None
        self._sam3_processor = None

        # SAM 3D config 路径
        if sam3d_config is None:
            self.sam3d_config = os.path.join(
                SAM3D_ROOT, "checkpoints/hf/pipeline.yaml"
            )
        else:
            self.sam3d_config = sam3d_config

    # ============== GPT Vision 自动物体检测 ==============

    # Read from the OPENAI_API_KEY environment variable (never hard-coded)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    def detect_objects_with_gpt(
        self,
        image: np.ndarray,
        model: str = "gpt-5.2",
        max_objects: int = 5
    ) -> List[str]:
        """
        使用 GPT Vision API 自动识别图像中的前景物体

        Args:
            image: RGB 图像 numpy 数组
            model: GPT 模型名称
            max_objects: 最大物体数

        Returns:
            物体名称列表（英文），如 ["red ball", "wooden chair"]
        """
        if not self.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or pass --text-prompts (with --no-gpt) "
                "to specify the objects manually.")

        print(f"  Calling GPT Vision API ({model}) to detect objects...")

        # 将图像编码为 base64
        pil_image = Image.fromarray(image)
        from io import BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 构造请求
        headers = {
            "Authorization": f"Bearer {self.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt = (
            f"List the names of ALL objects in this image. Try your best to identify every single object you can see. "
            f"Rules:\n"
            f"- Include every visible object, no matter how small or partially occluded\n"
            f"- Do NOT include background elements (floor, wall, sky, ground, table surface, etc.)\n"
            f"- Use generic, common category nouns only (e.g., 'car' instead of 'silver sedan', 'bear' instead of 'teddy bear', 'table' instead of 'wooden coffee table')\n"
            f"- Do NOT add adjectives like color, material, or brand. Keep it as simple as possible\n"
            f"- Maximum {max_objects} objects\n"
            f"- Return ONLY a JSON array of strings, nothing else\n"
            f"Example: [\"car\", \"cup\", \"bear\"]"
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_completion_tokens": 300
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        print(f"  GPT response: {content}")

        # 解析 JSON 数组
        # 处理可能的 markdown 代码块包裹
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]).strip()

        objects = json.loads(content)
        print(f"  GPT detected {len(objects)} objects: {objects}")
        return objects

    # ============== SAM3 文本提示分割 ==============

    def _load_sam3(self):
        """加载 SAM3 模型（用于文本提示分割）"""
        if self._sam3_model is not None:
            return

        print("  Loading SAM3...")
        from transformers import Sam3Processor, Sam3Model

        self._sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
        self._sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        print("  SAM3 loaded.")

    def _unload_sam3(self):
        """释放 SAM3 模型内存"""
        if self._sam3_model is not None:
            print("  Releasing SAM3 memory...")
            del self._sam3_model
            del self._sam3_processor
            self._sam3_model = None
            self._sam3_processor = None
            torch.cuda.empty_cache()

    def segment_with_sam3(
        self,
        image: np.ndarray,
        text_prompts: List[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_objects: int = 5
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        使用 SAM3 文本提示进行分割

        Args:
            image: RGB 图像 numpy 数组
            text_prompts: 文本提示列表，如 ["ball", "car", "person"]
            threshold: 检测阈值
            mask_threshold: mask 阈值
            max_objects: 最大物体数

        Returns:
            masks: mask 列表
            bboxes: 对应的边界框列表
        """
        self._load_sam3()

        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)
        H, W = image.shape[:2]

        all_masks = []
        all_bboxes = []

        for text_prompt in text_prompts:
            if len(all_masks) >= max_objects:
                break

            print(f"    Segmenting with text prompt: '{text_prompt}'")

            # 准备输入
            inputs = self._sam3_processor(
                images=pil_image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self._sam3_model(**inputs)

            # 后处理
            results = self._sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=[(H, W)]
            )[0]

            masks = results.get('masks', [])
            boxes = results.get('boxes', torch.tensor([]))
            scores = results.get('scores', torch.tensor([]))

            print(f"      Found {len(masks)} instances of '{text_prompt}'")

            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if len(all_masks) >= max_objects:
                    break

                mask_np = mask.cpu().numpy().astype(bool)
                box_np = box.cpu().numpy()

                all_masks.append(mask_np)
                all_bboxes.append({
                    'label': text_prompt,
                    'x0': int(box_np[0]),
                    'y0': int(box_np[1]),
                    'x1': int(box_np[2]),
                    'y1': int(box_np[3]),
                    'score': float(score.cpu().numpy())
                })
                print(f"        Instance {i+1}: score={score:.3f}, box=[{box_np[0]:.0f}, {box_np[1]:.0f}, {box_np[2]:.0f}, {box_np[3]:.0f}]")

        return all_masks, all_bboxes

    def detect_and_segment_with_sam3(
        self,
        image: np.ndarray,
        text_prompts: Optional[List[str]] = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_objects: int = 5,
        use_gpt: bool = True,
        gpt_model: str = "gpt-5.2"
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        使用 SAM3 自动检测并分割物体

        如果未提供 text_prompts 且 use_gpt=True，调用 GPT Vision 自动识别物体

        Args:
            image: RGB 图像
            text_prompts: 文本提示列表（可选）
            threshold: 检测阈值
            mask_threshold: mask 阈值
            max_objects: 最大物体数
            use_gpt: 是否使用 GPT Vision 自动检测
            gpt_model: GPT 模型名称

        Returns:
            masks: mask 列表
            bboxes: 对应的边界框列表
        """
        if text_prompts is None:
            # 调用 GPT Vision API 自动识别物体
            text_prompts = self.detect_objects_with_gpt(
                image, model=gpt_model, max_objects=max_objects
            )

        return self.segment_with_sam3(
            image,
            text_prompts=text_prompts,
            threshold=threshold,
            mask_threshold=mask_threshold,
            max_objects=max_objects
        )

    # ============== 调试输出辅助函数 ==============

    def _save_detection_visualization(
        self,
        image: np.ndarray,
        bboxes: List[Dict[str, Any]],
        output_path: str
    ):
        """
        保存检测结果可视化图片（原图 + BBox）

        Args:
            image: RGB 图像
            bboxes: BBox 列表
            output_path: 输出路径
        """
        img = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(img)
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
            # 绘制矩形框
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            # 绘制标签
            label = bbox.get('label', f'object_{i}')
            draw.text((x0 + 5, y0 + 5), label, fill=color)

        img.save(output_path)
        print(f"  Saved detection visualization: {output_path}")

    def _save_segmentation_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Dict[str, Any],
        output_path: str
    ):
        """
        保存 SAM 分割可视化图片（原图 + mask 叠加高亮）

        Args:
            image: RGB 图像
            mask: 分割 mask (bool)
            bbox: BBox 信息
            output_path: 输出路径
        """
        # 创建彩色 overlay
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 0]  # 绿色高亮

        # 混合原图和 mask
        alpha = 0.4
        result = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)

        # 绘制 bbox
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
        cv2.rectangle(result, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # 添加标签
        label = bbox.get('label', 'object')
        cv2.putText(result, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        Image.fromarray(result).save(output_path)

    def _save_foreground_cutout(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: str,
        background: str = 'white'
    ):
        """
        保存前景抠图（背景为白色或透明）

        Args:
            image: RGB 图像
            mask: 分割 mask (bool)
            output_path: 输出路径
            background: 'white' 或 'transparent'
        """
        if background == 'white':
            result = np.ones_like(image) * 255
            result[mask > 0] = image[mask > 0]
            Image.fromarray(result.astype(np.uint8)).save(output_path)
        else:  # transparent
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = (mask * 255).astype(np.uint8)
            Image.fromarray(rgba, 'RGBA').save(output_path)

    def _save_sam3d_input(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: str
    ):
        """
        保存 SAM 3D Objects 输入图片

        Args:
            image: RGB 图像
            mask: 分割 mask (bool)
            output_path: 输出路径
        """
        # SAM 3D 输入：前景保留，背景设为白色
        input_image = image.copy()
        input_image[mask == 0] = 255
        Image.fromarray(input_image.astype(np.uint8)).save(output_path)

    # ============== MoGe 深度与相机参数 ==============

    def _call_moge(self, image: np.ndarray) -> dict:
        """
        调用 MoGe 深度模型，返回完整输出（pointmap + intrinsics）

        带缓存：同一张图像只推理一次，后续调用直接返回缓存结果。

        Args:
            image: RGB 图像 numpy 数组 [H, W, 3] uint8

        Returns:
            dict: pointmap, depth, intrinsics, image_size
        """
        H, W = image.shape[:2]

        # 检查缓存（按图像尺寸匹配）
        if self._moge_cache is not None:
            cached_size = self._moge_cache.get("image_size")
            if cached_size == (H, W):
                return self._moge_cache

        self._load_sam3d()
        pipeline = self.sam3d_inference._pipeline

        # RGB numpy [H,W,3] uint8 → float32 tensor [3,H,W]
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).contiguous()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=pipeline.dtype):
                output = pipeline.depth_model(img_tensor)

        pointmap = output["pointmaps"]  # [H, W, 3]
        depth = pointmap[..., 2]        # Z 分量 = 深度
        intrinsics = output.get("intrinsics", None)

        # 如果 MoGe 未直接返回 intrinsics，从 pointmap 推断
        if intrinsics is None:
            from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
            intrinsics_result = infer_intrinsics_from_pointmap(
                pointmap.unsqueeze(0) if pointmap.dim() == 3 else pointmap,
                device=pointmap.device
            )
            intrinsics = intrinsics_result["intrinsics"]

        # 确保 intrinsics 是 2D [3, 3]
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.squeeze(0)

        result = {
            "pointmap": pointmap,
            "depth": depth,
            "intrinsics": intrinsics,
            "image_size": (H, W),
        }

        # 缓存结果
        self._moge_cache = result
        return result

    def estimate_depth(self, image: np.ndarray) -> torch.Tensor:
        """
        使用 SAM 3D 内部的 MoGe 估计全图深度

        Args:
            image: RGB 图像 numpy 数组 [H, W, 3] uint8

        Returns:
            depth: 深度图 tensor [H, W]，值越大越远
        """
        return self._call_moge(image)["depth"]

    def estimate_camera_params(self, image: np.ndarray) -> dict:
        """
        从 MoGe 深度模型提取相机内参

        MoGe 输出归一化 intrinsics（cx=cy=0.5），从中计算 FOV 和像素焦距。
        在 scene_posed 坐标系中，相机位于原点 (0,0,0)。

        坐标系约定:
        - MoGe 空间: X-right, Y-down, Z-forward (OpenCV)
        - scene_posed 空间: X-left, Y-up, Z-forward
          (PyTorch3D camera_to_pytorch3d_camera 翻转了 X)
        - 变换: p_scene = (-p_moge_x, -p_moge_y, p_moge_z)

        Args:
            image: RGB 图像 numpy 数组 [H, W, 3] uint8

        Returns:
            dict: 相机参数（intrinsics, FOV, 焦距, 位置, 朝向等）
        """
        import math

        moge_output = self._call_moge(image)
        intrinsics = moge_output["intrinsics"]  # [3, 3] 归一化
        H, W = moge_output["image_size"]

        fx_norm = intrinsics[0, 0].item()
        fy_norm = intrinsics[1, 1].item()
        cx_norm = intrinsics[0, 2].item()
        cy_norm = intrinsics[1, 2].item()

        # FOV (归一化坐标下)
        fov_x_rad = 2 * math.atan(0.5 / fx_norm)
        fov_y_rad = 2 * math.atan(0.5 / fy_norm)
        fov_x_deg = math.degrees(fov_x_rad)
        fov_y_deg = math.degrees(fov_y_rad)

        # 像素焦距
        fx_px = fx_norm * W
        fy_px = fy_norm * H

        camera_params = {
            "intrinsics": intrinsics.cpu().numpy().tolist(),
            "fov_x_deg": round(fov_x_deg, 2),
            "fov_y_deg": round(fov_y_deg, 2),
            "focal_length_px": [round(fx_px, 2), round(fy_px, 2)],
            "principal_point_norm": [round(cx_norm, 4), round(cy_norm, 4)],
            "image_width": W,
            "image_height": H,
            # 相机在 scene_posed 空间 (PyTorch3D) 中的位置
            "camera_position_scene": [0.0, 0.0, 0.0],
            "camera_look_direction_scene": [0.0, 0.0, 1.0],
            "camera_up_scene": [0.0, 1.0, 0.0],
            "coordinate_convention": {
                "scene_posed_space": "X-left, Y-up, Z-forward (Z increases away from camera)",
                "moge_space": "OpenCV: X-right, Y-down, Z-forward",
                "transform_moge_to_scene": "p_scene = (-p_moge_x, -p_moge_y, p_moge_z)"
            }
        }

        print(f"  Camera params: FOV={fov_x_deg:.1f}x{fov_y_deg:.1f} deg, "
              f"focal=[{fx_px:.0f}, {fy_px:.0f}]px")

        return camera_params

    def estimate_floor_y(
        self,
        image: np.ndarray,
        masks: list = None,
    ):
        """
        从 MoGe pointmap 提取地板在 scene_posed 空间中的 Y 坐标

        算法:
        1. 获取 MoGe pointmap（从缓存）
        2. 排除物体 mask 区域，聚焦图像下半部分（地板通常在画面下方）
        3. 提取候选像素的 MoGe Y 坐标（Y-down 系统中地板 Y 值较大）
        4. 取第 80~98 百分位的中位数作为地板 Y
        5. 转换: floor_y_scene = -floor_y_moge

        Args:
            image: RGB 图像 numpy 数组 [H, W, 3] uint8
            masks: 前景物体 mask 列表（用于排除物体区域）

        Returns:
            float or None: 地板在 scene_posed 空间中的 Y 坐标，
                          如果无法确定则返回 None
        """
        moge_output = self._call_moge(image)
        pointmap = moge_output["pointmap"]  # [H, W, 3] tensor, MoGe 相机坐标系
        H, W = moge_output["image_size"]

        pointmap_np = pointmap.cpu().numpy()  # [H, W, 3]

        # 构建非物体 mask（True = 可用的背景/地板像素）
        available_mask = np.ones((H, W), dtype=bool)
        if masks is not None:
            for mask in masks:
                if mask.shape[:2] == (H, W):
                    available_mask &= (mask == 0)

        # 排除 pointmap 中的无效点（零值或 inf）
        valid_mask = np.isfinite(pointmap_np).all(axis=2)
        point_norm = np.linalg.norm(pointmap_np, axis=2)
        valid_mask &= (point_norm > 1e-6)

        # 聚焦图像下半部分（地板通常在画面下方）
        min_pixels = 100  # 最少需要的候选像素数

        for bottom_ratio in [0.50, 0.70, 1.0]:
            row_start = int(H * (1.0 - bottom_ratio))
            region_mask = np.zeros((H, W), dtype=bool)
            region_mask[row_start:, :] = True

            candidate_mask = available_mask & valid_mask & region_mask
            n_candidates = candidate_mask.sum()

            if n_candidates >= min_pixels:
                break

        if n_candidates < min_pixels:
            print(f"  Floor Y estimation: insufficient pixels ({n_candidates}), skipping")
            return None

        # 提取候选像素的 MoGe Y 坐标
        # MoGe 坐标系: Y-down，所以地板的 Y 值较大（图像下方 = 更大的 Y）
        moge_y_values = pointmap_np[candidate_mask, 1]

        # 取第 80~98 百分位的中位数（排除极端噪声，聚焦地板区域）
        p80 = np.percentile(moge_y_values, 80)
        p98 = np.percentile(moge_y_values, 98)
        floor_band = moge_y_values[(moge_y_values >= p80) & (moge_y_values <= p98)]

        if len(floor_band) < 10:
            print(f"  Floor Y estimation: too few points in floor band ({len(floor_band)}), skipping")
            return None

        floor_y_moge = float(np.median(floor_band))

        # 转换到 scene_posed 空间: p_scene = (-p_moge_x, -p_moge_y, p_moge_z)
        floor_y_scene = -floor_y_moge

        # 合理性检查：floor_y_scene 应该是负值（地板在相机下方）
        # 在 scene_posed Y-up 系统中，地板通常 Y < 0（相机在地板上方）
        if floor_y_scene > 0.5:
            print(f"  Floor Y estimation: floor_y_scene={floor_y_scene:.4f} seems too high, skipping")
            return None

        print(f"  Floor Y (MoGe): {floor_y_moge:.4f} → Floor Y (scene_posed): {floor_y_scene:.4f}"
              f" (from {len(floor_band)} pixels, bottom {bottom_ratio*100:.0f}%)")

        return floor_y_scene

    def estimate_ground_plane(
        self,
        image: np.ndarray,
        masks: list = None,
    ):
        """
        从 MoGe pointmap 用 RANSAC 拟合地面平面

        地面在 3D 空间中是一个倾斜平面（透视效果），用单一 floor_y 会导致
        远处物体被误认为悬浮。RANSAC 平面拟合可以正确处理这种情况。

        Args:
            image: RGB 图像 numpy 数组 [H, W, 3] uint8
            masks: 前景物体 mask 列表（用于排除物体区域）

        Returns:
            dict or None: {
                "normal": [nx, ny, nz],  # scene_posed 空间平面法向量 (Y-up)
                "d": float,              # 平面方程 normal·p + d = 0
            }
        """
        moge_output = self._call_moge(image)
        pointmap = moge_output["pointmap"]  # [H, W, 3] MoGe 相机坐标
        H, W = moge_output["image_size"]

        pointmap_np = pointmap.cpu().numpy()  # [H, W, 3]

        # 构建非物体 mask
        available_mask = np.ones((H, W), dtype=bool)
        if masks is not None:
            for mask in masks:
                if mask.shape[:2] == (H, W):
                    available_mask &= (mask == 0)

        # 有效点 mask
        valid_mask = np.isfinite(pointmap_np).all(axis=2)
        point_norm = np.linalg.norm(pointmap_np, axis=2)
        valid_mask &= (point_norm > 1e-6)

        # 聚焦图像下半部分（地面区域）
        min_pixels = 200
        for bottom_ratio in [0.50, 0.70, 1.0]:
            row_start = int(H * (1.0 - bottom_ratio))
            region_mask = np.zeros((H, W), dtype=bool)
            region_mask[row_start:, :] = True
            candidate_mask = available_mask & valid_mask & region_mask
            n_candidates = candidate_mask.sum()
            if n_candidates >= min_pixels:
                break

        if n_candidates < min_pixels:
            print(f"  Ground plane estimation: insufficient pixels ({n_candidates})")
            return None

        # 提取候选 3D 点，转换到 scene_posed 坐标系
        moge_pts = pointmap_np[candidate_mask]  # (N, 3) MoGe: X-right, Y-down, Z-forward
        # scene_posed: X-left, Y-up, Z-forward
        scene_pts = np.column_stack([-moge_pts[:, 0], -moge_pts[:, 1], moge_pts[:, 2]])

        # 预过滤：用 Y 值的百分位排除非地面点（墙壁等）
        # 在 scene_posed Y-up 中，地面 Y 较小（负值）
        y_vals = scene_pts[:, 1]
        p10 = np.percentile(y_vals, 10)
        p60 = np.percentile(y_vals, 60)
        ground_band_mask = (y_vals >= p10) & (y_vals <= p60)
        ground_candidates = scene_pts[ground_band_mask]

        if len(ground_candidates) < 100:
            print(f"  Ground plane estimation: too few ground candidates ({len(ground_candidates)})")
            return None

        # RANSAC 平面拟合
        best_normal = None
        best_d = None
        best_inlier_count = 0
        n_points = len(ground_candidates)
        n_iterations = 200
        distance_threshold = 0.02  # 2cm 内点阈值

        rng = np.random.RandomState(42)
        for _ in range(n_iterations):
            # 随机采样 3 个点
            idx = rng.choice(n_points, 3, replace=False)
            p0, p1, p2 = ground_candidates[idx]

            # 计算平面法向量
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len

            # 确保法向量朝上 (Y-up: normal[1] > 0)
            if normal[1] < 0:
                normal = -normal

            # 计算 d: normal · p + d = 0
            d = -np.dot(normal, p0)

            # 计算所有点到平面的距离
            distances = np.abs(ground_candidates @ normal + d)
            inlier_count = (distances < distance_threshold).sum()

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_normal = normal
                best_d = d

        if best_normal is None or best_inlier_count < 50:
            print(f"  Ground plane estimation: RANSAC failed "
                  f"(best inliers: {best_inlier_count})")
            return None

        # 用所有内点重新拟合（least squares refinement）
        distances = np.abs(ground_candidates @ best_normal + best_d)
        inlier_mask = distances < distance_threshold
        inlier_points = ground_candidates[inlier_mask]

        centroid = inlier_points.mean(axis=0)
        centered = inlier_points - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        refined_normal = Vt[2]  # 最小奇异值对应的方向
        if refined_normal[1] < 0:
            refined_normal = -refined_normal
        refined_d = -np.dot(refined_normal, centroid)

        inlier_ratio = best_inlier_count / len(ground_candidates)
        print(f"  Ground plane RANSAC: {best_inlier_count}/{len(ground_candidates)} "
              f"inliers ({inlier_ratio:.1%}), normal={refined_normal.round(4).tolist()}")

        result = {
            "normal": refined_normal.tolist(),
            "d": float(refined_d),
        }
        return result

    def filter_masks_by_depth(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[Dict[str, Any]],
        max_depth_ratio: float = 2.0
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]], List[int]]:
        """
        基于 MoGe 深度过滤远景物体，只保留近景

        Args:
            image: RGB 图像 numpy 数组
            masks: mask 列表
            bboxes: BBox 列表
            max_depth_ratio: 最大深度倍数（相对于最近物体）

        Returns:
            filtered_masks: 过滤后的 mask 列表
            filtered_bboxes: 过滤后的 bbox 列表
            kept_indices: 通过过滤的原始索引列表
        """
        if len(masks) <= 1:
            return masks, bboxes, list(range(len(masks)))

        print("  Estimating depth with MoGe...")
        depth_map = self.estimate_depth(image)

        # 计算每个 mask 区域的中位深度
        mask_depths = []
        for i, mask in enumerate(masks):
            mask_tensor = torch.from_numpy(mask).to(depth_map.device)
            masked_depth = depth_map[mask_tensor > 0]
            # 使用绝对值处理可能的负深度
            median_d = masked_depth.abs().median().item()
            mask_depths.append(median_d)
            label = bboxes[i].get('label', f'obj_{i}') if i < len(bboxes) else f'obj_{i}'
            print(f"    {label}: median depth = {median_d:.4f}")

        # 以最近物体为基准计算阈值
        min_depth = min(mask_depths)
        threshold = min_depth * max_depth_ratio
        print(f"  Nearest depth: {min_depth:.4f}, threshold: {threshold:.4f} (ratio={max_depth_ratio})")

        # 过滤
        filtered_masks = []
        filtered_bboxes = []
        kept_indices = []
        for i, (mask, bbox, d) in enumerate(zip(masks, bboxes, mask_depths)):
            label = bbox.get('label', f'obj_{i}')
            if d <= threshold:
                filtered_masks.append(mask)
                filtered_bboxes.append(bbox)
                kept_indices.append(i)
                print(f"    KEEP '{label}' (depth={d:.4f})")
            else:
                print(f"    SKIP '{label}' (depth={d:.4f} > {threshold:.4f})")

        return filtered_masks, filtered_bboxes, kept_indices

    # ============== SAM 3D Objects 重建 ==============

    def _load_sam3d(self):
        """加载 SAM 3D Objects 模型"""
        if self.sam3d_inference is not None:
            return

        print("  Loading SAM 3D Objects...")

        # 导入 inference 模块
        from inference import Inference

        self.sam3d_inference = Inference(
            self.sam3d_config,
            compile=self.compile_model
        )
        print("  SAM 3D Objects loaded.")

    def reconstruct_objects(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        seed: int = 42,
        output_dir: Optional[str] = None,
        bboxes: Optional[List[Dict[str, Any]]] = None,
        layout_optimize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        对每个 mask 进行 3D 重建

        Args:
            image: RGB 图像
            masks: mask 列表
            seed: 随机种子
            output_dir: 输出目录（用于保存调试图片）
            bboxes: BBox 列表（用于标签）
            layout_optimize: 是否启用 layout post-optimization（默认开启，三阶段优化 pose）

        Returns:
            outputs 列表，每个包含 gaussian, mesh, rotation, translation, scale 等
        """
        self._load_sam3d()

        outputs = []
        for i, mask in enumerate(masks):
            label = "object"
            if bboxes and i < len(bboxes):
                label = bboxes[i].get('label', f'object_{i}')

            opt_tag = " [+layout_optimize]" if layout_optimize else ""
            print(f"    Reconstructing {label} ({i+1}/{len(masks)}){opt_tag}...")

            # 保存 SAM 3D 输入图片
            if output_dir:
                sam3d_input_path = os.path.join(output_dir, f"sam3d_input_{i}.png")
                self._save_sam3d_input(image, mask, sam3d_input_path)
                print(f"      Saved SAM 3D input: {sam3d_input_path}")

            try:
                output = self.sam3d_inference(
                    image, mask, seed=seed,
                    with_layout_postprocess=layout_optimize
                )
                output['mask_index'] = i  # 记录此 output 对应的 mask 索引
                outputs.append(output)
                n_gs = output['gaussian'][0].get_xyz.shape[0]
                has_mesh = 'mesh' in output and output['mesh'] is not None
                mesh_info = ""
                if has_mesh:
                    mesh_obj = output['mesh'][0]
                    mesh_info = f", Mesh: {mesh_obj.vertices.shape[0]} verts, {mesh_obj.faces.shape[0]} faces"
                print(f"      Success! Gaussian: {n_gs} points{mesh_info}")
            except Exception as e:
                print(f"      Error: {e}, skipping {label}")
                import traceback
                traceback.print_exc()

        return outputs

    # ============== 场景合并 ==============

    def merge_to_scene(self, outputs: List[Dict[str, Any]]):
        """
        合并所有物体到统一场景

        Args:
            outputs: 3D 重建结果列表

        Returns:
            合并后的 scene_gs
        """
        from inference import make_scene
        return make_scene(*outputs)

    # ============== 主运行流程 ==============

    def run(
        self,
        image_path: str,
        output_dir: str = "output",
        max_objects: int = 5,
        seed: int = 42,
        save_intermediate: bool = True,
        save_gif: bool = True,
        gif_resolution: int = 512,
        gif_frames: int = 120,
        text_prompts: Optional[List[str]] = None,
        sam3_threshold: float = 0.5,
        use_gpt: bool = True,
        gpt_model: str = "gpt-5.2",
        max_depth_ratio: float = 2.0,
        no_depth_filter: bool = False,
        layout_optimize: bool = True
    ) -> Any:
        """
        运行完整 Pipeline

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            max_objects: 最大物体数
            seed: 随机种子
            save_intermediate: 是否保存中间结果
            save_gif: 是否生成旋转 GIF
            gif_resolution: GIF 分辨率
            gif_frames: GIF 帧数
            text_prompts: SAM3 文本提示列表
            sam3_threshold: SAM3 检测阈值
            use_gpt: 是否使用 GPT Vision 自动检测物体
            gpt_model: GPT 模型名称
            max_depth_ratio: 最大深度倍数（相对于最近物体），默认 2.0
            no_depth_filter: 是否禁用深度过滤
            layout_optimize: 是否启用 layout post-optimization（默认开启）

        Returns:
            scene_gs: 合并后的 Gaussian Splat 场景
        """
        os.makedirs(output_dir, exist_ok=True)
        total_steps = 4

        # ========== Step 1: 加载图像 ==========
        print(f"\n[1/{total_steps}] Loading image: {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        H, W = image.shape[:2]
        print(f"  Image size: {W}x{H}")

        # ========== Step 2: SAM3 检测和分割物体 ==========
        if text_prompts:
            print(f"\n[2/{total_steps}] Segmenting objects with SAM3 (manual prompts: {text_prompts})...")
        else:
            print(f"\n[2/{total_steps}] Detecting objects with GPT Vision ({gpt_model}) + segmenting with SAM3...")
        masks, bboxes = self.detect_and_segment_with_sam3(
            image,
            text_prompts=text_prompts,
            threshold=sam3_threshold,
            mask_threshold=sam3_threshold,
            max_objects=max_objects,
            use_gpt=use_gpt,
            gpt_model=gpt_model
        )
        print(f"  SAM3 detected {len(masks)} objects")

        # 保存检测结果
        if save_intermediate and len(bboxes) > 0:
            detection_data = {"objects": bboxes, "mode": "sam3"}
            if not text_prompts:
                detection_data["gpt_model"] = gpt_model
                detection_data["gpt_auto_detect"] = True
            detection_path = os.path.join(output_dir, "detection_result.json")
            with open(detection_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
            print(f"  Saved detection result to: {detection_path}")

            detection_vis_path = os.path.join(output_dir, "detection_visualization.png")
            self._save_detection_visualization(image, bboxes, detection_vis_path)

        # 释放 SAM3 内存
        self._unload_sam3()

        print(f"  Got {len(masks)} masks")

        # 保存 masks 和调试输出
        if save_intermediate and len(masks) > 0:
            for i, mask in enumerate(masks):
                # 获取对应的 bbox（如果有）
                bbox = bboxes[i] if (bboxes and i < len(bboxes)) else {
                    'x0': 0, 'y0': 0, 'x1': W, 'y1': H, 'label': f'object_{i}'
                }

                # 1. 保存二值 mask
                mask_path = os.path.join(output_dir, f"mask_{i}.png")
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                print(f"  Saved mask: {mask_path}")

                # 2. 保存分割可视化（原图 + mask 叠加）
                seg_vis_path = os.path.join(output_dir, f"sam_segmentation_{i}.png")
                self._save_segmentation_visualization(image, mask, bbox, seg_vis_path)
                print(f"  Saved segmentation visualization: {seg_vis_path}")

                # 3. 保存前景抠图（白色背景）
                fg_path = os.path.join(output_dir, f"sam_foreground_{i}.png")
                self._save_foreground_cutout(image, mask, fg_path, background='white')
                print(f"  Saved foreground cutout: {fg_path}")

        if len(masks) == 0:
            print("  Error: No objects segmented!")
            return None

        # ========== Step 2.5: 深度过滤 ==========
        # kept_indices: 记录通过过滤的原始检测索引（用于后续 reconstruction mapping）
        kept_indices = list(range(len(masks)))
        if not no_depth_filter and len(masks) > 1:
            print(f"\n[2.5/{total_steps}] Filtering objects by depth (max_depth_ratio={max_depth_ratio})...")
            masks, bboxes, kept_indices = self.filter_masks_by_depth(
                image, masks, bboxes,
                max_depth_ratio=max_depth_ratio
            )
            print(f"  After depth filtering: {len(masks)} objects remain")

            if len(masks) == 0:
                print("  Error: All objects filtered out! Try increasing --max-depth-ratio")
                return None

        # ========== Step 3: 3D 重建 ==========
        opt_msg = " (with layout optimization)" if layout_optimize else ""
        print(f"\n[3/{total_steps}] Reconstructing with SAM 3D Objects{opt_msg}...")
        outputs = self.reconstruct_objects(
            image,
            masks,
            seed=seed,
            output_dir=output_dir if save_intermediate else None,
            bboxes=bboxes,
            layout_optimize=layout_optimize
        )

        if len(outputs) == 0:
            print("  Error: No objects reconstructed!")
            return None

        # 保存单个物体
        if save_intermediate:
            for i, out in enumerate(outputs):
                obj_path = os.path.join(output_dir, f"object_{i}.ply")
                out["gs"].save_ply(obj_path)
                print(f"  Saved: {obj_path}")

                # 保存位姿信息
                pose_path = os.path.join(output_dir, f"object_{i}_pose.json")
                pose_info = {
                    "rotation": out["rotation"].cpu().numpy().tolist(),
                    "translation": out["translation"].cpu().numpy().tolist(),
                    "scale": out["scale"].cpu().numpy().tolist()
                }
                with open(pose_path, 'w') as f:
                    json.dump(pose_info, f, indent=2)

                # 保存 Mesh（局部坐标 + 场景坐标）
                if 'mesh' in out and out['mesh'] is not None:
                    try:
                        from mesh_to_particles import save_mesh_ply, transform_mesh_to_scene
                        import trimesh as _trimesh

                        mesh_result = out['mesh'][0]

                        # 局部坐标 mesh (Z-up, 原始输出)
                        mesh_local_path = os.path.join(output_dir, f"object_{i}_mesh.ply")
                        save_mesh_ply(mesh_result, mesh_local_path)
                        print(f"  Saved mesh (local): {mesh_local_path}")

                        # 场景坐标 mesh (Y-up, pose 变换后)
                        verts_scene, faces = transform_mesh_to_scene(mesh_result, out)
                        mesh_posed = _trimesh.Trimesh(
                            vertices=verts_scene, faces=faces, process=False)
                        mesh_posed_path = os.path.join(output_dir, f"object_{i}_mesh_posed.ply")
                        mesh_posed.export(mesh_posed_path)
                        print(f"  Saved mesh (posed): {mesh_posed_path}")
                    except Exception as e:
                        print(f"  Warning: Failed to save mesh for object_{i}: {e}")

        # 保存 reconstruction mapping（.ply 索引 → 原始检测索引）
        if save_intermediate:
            reconstruction_map = []
            for ply_idx, out in enumerate(outputs):
                mask_idx = out.get('mask_index', ply_idx)
                det_idx = kept_indices[mask_idx] if mask_idx < len(kept_indices) else mask_idx
                reconstruction_map.append(det_idx)

            map_path = os.path.join(output_dir, "reconstruction_map.json")
            # 读取原始检测数（从 detection_result.json 中获取）
            detection_path_tmp = os.path.join(output_dir, "detection_result.json")
            n_originally_detected = len(bboxes)
            if os.path.exists(detection_path_tmp):
                with open(detection_path_tmp, 'r') as f:
                    n_originally_detected = len(json.load(f).get('objects', []))
            with open(map_path, 'w') as f:
                json.dump({
                    "ply_to_detection": reconstruction_map,
                    "n_detected": n_originally_detected,
                    "n_reconstructed": len(outputs)
                }, f, indent=2)
            print(f"  Saved reconstruction map: {map_path} "
                  f"({len(outputs)} reconstructed from {len(kept_indices)} filtered)")

            # 保存 mesh 元数据
            mesh_info_list = []
            for idx, out in enumerate(outputs):
                mesh_entry = {"index": idx, "mesh_ply": f"object_{idx}_mesh_posed.ply"}
                if 'mesh' in out and out['mesh'] is not None:
                    m = out['mesh'][0]
                    mesh_entry["n_vertices"] = int(m.vertices.shape[0])
                    mesh_entry["n_faces"] = int(m.faces.shape[0])
                    mesh_entry["has_mesh"] = True
                else:
                    mesh_entry["has_mesh"] = False
                mesh_info_list.append(mesh_entry)
            mesh_info_path = os.path.join(output_dir, "mesh_info.json")
            with open(mesh_info_path, 'w') as f:
                json.dump({"n_objects": len(outputs), "objects": mesh_info_list}, f, indent=2)
            print(f"  Saved mesh info: {mesh_info_path}")

            # 更新 detection_result.json → 仅保留重建成功的物体
            detection_path = os.path.join(output_dir, "detection_result.json")
            if os.path.exists(detection_path):
                with open(detection_path, 'r') as f:
                    det_data = json.load(f)
                original_objects = det_data.get('objects', [])
                det_data['original_objects'] = original_objects  # 保留原始完整列表
                det_data['objects'] = [original_objects[di] for di in reconstruction_map
                                       if di < len(original_objects)]
                with open(detection_path, 'w') as f:
                    json.dump(det_data, f, indent=2)
                print(f"  Updated detection_result.json: {len(original_objects)} → {len(det_data['objects'])} objects")

        # ========== Step 4: 合并场景 + 归一化 + 保存 ==========
        print(f"\n[4/{total_steps}] Merging scene...")
        scene_gs = self.merge_to_scene(outputs)

        # 保存带位姿的场景
        posed_path = os.path.join(output_dir, "scene_posed.ply")
        scene_gs.save_ply(posed_path)
        print(f"  Saved posed scene: {posed_path}")

        # 提取并保存相机参数
        print(f"\n  Extracting camera parameters from MoGe...")
        try:
            camera_params = self.estimate_camera_params(image)

            # RANSAC 地面平面拟合（替代旧的 floor_y 单值估计）
            print(f"  Estimating ground plane from MoGe pointmap (RANSAC)...")
            try:
                ground_plane = self.estimate_ground_plane(image, masks=masks)
                if ground_plane is not None:
                    camera_params["ground_plane"] = ground_plane
                    # 向后兼容: 也计算 floor_y_scene (用旧方法)
                    floor_y = self.estimate_floor_y(image, masks=masks)
                    if floor_y is not None:
                        camera_params["floor_y_scene"] = round(float(floor_y), 6)
                    print(f"  Ground plane saved: normal={[round(n,4) for n in ground_plane['normal']]}")
                else:
                    print(f"  Ground plane estimation failed, falling back to floor_y...")
                    floor_y = self.estimate_floor_y(image, masks=masks)
                    if floor_y is not None:
                        camera_params["floor_y_scene"] = round(float(floor_y), 6)
                    else:
                        print(f"  Floor Y also not available")
            except Exception as e_floor:
                print(f"  Warning: Failed to estimate ground plane: {e_floor}")
                # 回退到旧方法
                try:
                    floor_y = self.estimate_floor_y(image, masks=masks)
                    if floor_y is not None:
                        camera_params["floor_y_scene"] = round(float(floor_y), 6)
                except Exception:
                    pass

            camera_params_path = os.path.join(output_dir, "camera_params.json")
            with open(camera_params_path, 'w') as f:
                json.dump(camera_params, f, indent=2)
            print(f"  Saved camera params: {camera_params_path}")
        except Exception as e:
            print(f"  Warning: Failed to extract camera params: {e}")
            camera_params = None

        # 归一化并保存
        print(f"  Normalizing...")
        from inference import ready_gaussian_for_video_rendering
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        scene_path = os.path.join(output_dir, "scene.ply")
        scene_gs.save_ply(scene_path)
        print(f"  Saved normalized scene: {scene_path}")

        # 生成 GIF
        if save_gif:
            print("  Rendering video...")
            try:
                import imageio
                from inference import render_video

                torch.cuda.empty_cache()

                video = render_video(
                    scene_gs,
                    r=1,
                    fov=60,
                    resolution=gif_resolution,
                    num_frames=gif_frames
                )["color"]

                gif_path = os.path.join(output_dir, "scene.gif")
                imageio.mimsave(
                    gif_path,
                    video,
                    format="GIF",
                    duration=1000/30,
                    loop=0
                )
                print(f"  Saved GIF: {gif_path}")
            except Exception as e:
                print(f"  Warning: Failed to generate GIF: {e}")
                import traceback
                traceback.print_exc()

        # 清理 MoGe 缓存
        self._moge_cache = None

        print(f"\n{'='*50}")
        print(f"Done! Output saved to: {output_dir}")
        print(f"  - scene.ply: Normalized Gaussian Splat scene")
        print(f"  - scene_posed.ply: Scene with original poses")
        if save_gif:
            print(f"  - scene.gif: 360-degree rotation animation")
        print(f"{'='*50}")

        return scene_gs


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Multi-Object Scene Reconstruction Pipeline"
    )
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--max-objects", type=int, default=5,
                        help="Maximum number of objects (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-intermediate", action="store_true",
                        help="Don't save intermediate results")
    parser.add_argument("--no-gif", action="store_true",
                        help="Don't generate rotation GIF")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--compile", action="store_true",
                        help="Compile SAM 3D model (slower startup, faster inference)")
    parser.add_argument("--gif-resolution", type=int, default=512,
                        help="GIF resolution (default: 512)")
    parser.add_argument("--gif-frames", type=int, default=120,
                        help="GIF frame count (default: 120)")

    # SAM3 相关参数
    parser.add_argument("--text-prompts", type=str, default=None,
                        help="Comma-separated text prompts for SAM3 (e.g., 'ball,cup,toy'). If not specified, uses GPT Vision to auto-detect")
    parser.add_argument("--sam3-threshold", type=float, default=0.5,
                        help="SAM3 detection threshold (default: 0.5)")

    # GPT Vision 相关参数
    parser.add_argument("--no-gpt", action="store_true",
                        help="Disable GPT Vision auto-detection (requires --text-prompts)")
    parser.add_argument("--gpt-model", type=str, default="gpt-5.2",
                        help="GPT model for vision detection (default: gpt-5.2)")

    # 深度过滤参数
    parser.add_argument("--max-depth-ratio", type=float, default=2.0,
                        help="Max depth ratio relative to nearest object (default: 2.0). "
                             "Objects deeper than nearest_depth * ratio will be skipped.")
    parser.add_argument("--no-depth-filter", action="store_true",
                        help="Disable depth filtering (reconstruct all detected objects)")

    # Layout post-optimization 参数（默认开启）
    parser.add_argument("--no-layout-optimize", action="store_true",
                        help="Disable layout post-optimization (enabled by default)")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"Error: Input file not found: {args.image}")
        sys.exit(1)

    # 检查参数一致性
    if args.no_gpt and not args.text_prompts:
        print("Error: --no-gpt requires --text-prompts to specify objects manually")
        sys.exit(1)
    if not args.text_prompts and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Export it for GPT Vision detection, "
              "or pass --text-prompts (optionally with --no-gpt)")
        sys.exit(1)

    # 解析文本提示
    text_prompts = None
    if args.text_prompts:
        text_prompts = [p.strip() for p in args.text_prompts.split(",")]
        print(f"Using manual text prompts: {text_prompts}")
    else:
        print(f"Using GPT Vision ({args.gpt_model}) for auto object detection")

    # 创建 Pipeline
    pipeline = AutoScenePipeline(
        device=args.device,
        compile_model=args.compile
    )

    # 运行完整 pipeline
    pipeline.run(
        args.image,
        output_dir=args.output_dir,
        max_objects=args.max_objects,
        seed=args.seed,
        save_intermediate=not args.no_intermediate,
        save_gif=not args.no_gif,
        gif_resolution=args.gif_resolution,
        gif_frames=args.gif_frames,
        text_prompts=text_prompts,
        sam3_threshold=args.sam3_threshold,
        use_gpt=not args.no_gpt,
        gpt_model=args.gpt_model,
        max_depth_ratio=args.max_depth_ratio,
        no_depth_filter=args.no_depth_filter,
        layout_optimize=not args.no_layout_optimize
    )


if __name__ == "__main__":
    main()
