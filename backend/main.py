import base64
import io
import os
import time
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    from transformers import DeiTForImageClassificationWithTeacher
except ImportError:
    DeiTForImageClassificationWithTeacher = None

VIT = os.getenv("VIT_MODEL_ID", "google/vit-base-patch16-224")
DEIT = os.getenv("DEIT_MODEL_ID", "facebook/deit-base-distilled-patch16-224")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: dict[str, dict[str, Any]] = {}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(
    title="Vision Transformer & DeiT Learning Studio API",
    description="Backend académique pour l'analyse en profondeur des architectures ViT et DeiT",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load(key: str) -> dict[str, Any]:
    """Charge et met en cache le processeur et le modèle demandé avec eager attention."""
    if key not in ("vit", "deit"):
        raise ValueError("Le modèle doit être 'vit' ou 'deit'.")

    if key not in MODELS:
        model_id = VIT if key == "vit" else DEIT
        processor = AutoImageProcessor.from_pretrained(model_id)

        if key == "deit" and DeiTForImageClassificationWithTeacher is not None:
            model = DeiTForImageClassificationWithTeacher.from_pretrained(
                model_id, attn_implementation="eager"
            )
        else:
            model = AutoModelForImageClassification.from_pretrained(
                model_id, attn_implementation="eager"
            )

        model.to(DEVICE)
        model.eval()
        MODELS[key] = {
            "processor": processor,
            "model": model,
            "id": model_id,
        }

    return MODELS[key]


def read_image(raw: bytes) -> Image.Image:
    """Ouvre une image reçue et la convertit en RGB."""
    try:
        img = Image.open(io.BytesIO(raw))
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Fichier image invalide.")


def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def pil_to_base64_png(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def preprocess_image(image: Image.Image, mode: str = "resize_crop_256") -> Image.Image:
    """
    Applique le pré-traitement ImageNet standard:
    - 'resize_crop_256': resize bicubique à 256x256 puis center crop 224x224
    - 'direct_224': resize direct 224x224
    """
    if mode == "resize_crop_256":
        resized = image.resize((256, 256), Image.Resampling.BICUBIC)
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        return resized.crop((left, top, left + 224, top + 224))
    else:
        return image.resize((224, 224), Image.Resampling.BICUBIC)


def infer_single(bundle: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    """Exécute une inférence PyTorch et extrait les top-5 classes."""
    processor = bundle["processor"]
    model = bundle["model"]
    model_id = bundle["id"]

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    elapsed_ms = (time.perf_counter() - start) * 1000
    values, indices = torch.topk(probabilities, min(5, len(probabilities)))

    labels = model.config.id2label

    def get_label(idx: int) -> str:
        return str(labels.get(int(idx), labels.get(str(int(idx)), f"Classe {idx}")))

    top5 = [
        {
            "class_id": int(idx.item()),
            "label": get_label(int(idx.item())),
            "probability": round(float(val.item()), 4),
        }
        for val, idx in zip(values, indices)
    ]

    return {
        "model_id": model_id,
        "predicted_class_id": top5[0]["class_id"],
        "prediction": top5[0]["label"],
        "confidence": top5[0]["probability"],
        "top5": top5,
        "inference_ms": round(elapsed_ms, 2),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(MODELS.keys()),
        "models": {"vit": VIT, "deit": DEIT},
    }


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    return {
        "vit": {
            "id": VIT,
            "architecture": "Vision Transformer (ViT-Base/16)",
            "patch_size": 16,
            "image_size": 224,
            "num_patches": 196,
            "hidden_size": 768,
            "layers": 12,
            "heads": 12,
            "head_dim": 64,
            "mlp_dim": 3072,
            "tokens": 197,
            "params": "~86M",
            "activation": "GELU",
            "distillation": False,
        },
        "deit": {
            "id": DEIT,
            "architecture": "Data-efficient Image Transformer (DeiT-Base Distilled)",
            "patch_size": 16,
            "image_size": 224,
            "num_patches": 196,
            "hidden_size": 768,
            "layers": 12,
            "heads": 12,
            "head_dim": 64,
            "mlp_dim": 3072,
            "tokens": 198,
            "params": "~86M",
            "activation": "GELU",
            "distillation": True,
            "teacher": "RegNetY-160 (CNN)",
            "heads_count": 2,  # CLS head + DIST head
        },
    }


@app.post("/api/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    mode: str = Query("resize_crop_256", pattern="^(resize_crop_256|direct_224)$"),
    patch_idx: int = Query(86, ge=0, le=195),
) -> dict[str, Any]:
    """
    Effectue le pré-traitement réel de l'image :
    - Avant / après
    - Canaux R, G, B réels (avec coloration rouge, verte, bleue)
    - Histogrammes RGB
    - Normalisation ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    - Échantillon de patch extrait
    """
    raw = await file.read()
    orig_img = read_image(raw)
    orig_w, orig_h = orig_img.size

    proc_img = preprocess_image(orig_img, mode=mode)
    arr = np.array(proc_img, dtype=np.float32) / 255.0  # (224, 224, 3)

    # Création des images de canaux colorés réels
    # Canal Rouge pur (R, 0, 0)
    red_arr = np.zeros_like(arr, dtype=np.uint8)
    red_arr[:, :, 0] = (arr[:, :, 0] * 255).astype(np.uint8)
    red_img = Image.fromarray(red_arr)

    # Canal Vert pur (0, G, 0)
    green_arr = np.zeros_like(arr, dtype=np.uint8)
    green_arr[:, :, 1] = (arr[:, :, 1] * 255).astype(np.uint8)
    green_img = Image.fromarray(green_arr)

    # Canal Bleu pur (0, 0, B)
    blue_arr = np.zeros_like(arr, dtype=np.uint8)
    blue_arr[:, :, 2] = (arr[:, :, 2] * 255).astype(np.uint8)
    blue_img = Image.fromarray(blue_arr)

    # Calcul des histogrammes réels (32 bins de 0 à 1)
    hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0.0, 1.0))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0.0, 1.0))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0.0, 1.0))

    # Tenseur normalisé ImageNet
    norm_tensor = (arr - IMAGENET_MEAN) / IMAGENET_STD  # shape: (224, 224, 3)
    norm_tensor_chfirst = np.transpose(norm_tensor, (2, 0, 1))  # (3, 224, 224)

    # Extraction du patch demandé (index 0..195)
    row = patch_idx // 14
    col = patch_idx % 14
    y_start, y_end = row * 16, (row + 1) * 16
    x_start, x_end = col * 16, (col + 1) * 16
    patch_crop = proc_img.crop((x_start, y_start, x_end, y_end))
    patch_raw_pixels = (np.array(patch_crop)).tolist()  # 16x16x3

    return {
        "original": {
            "width": orig_w,
            "height": orig_h,
            "preview": pil_to_base64_jpeg(orig_img),
        },
        "processed": {
            "width": 224,
            "height": 224,
            "mode": mode,
            "preview": pil_to_base64_jpeg(proc_img),
        },
        "channels": {
            "red_preview": pil_to_base64_png(red_img),
            "green_preview": pil_to_base64_png(green_img),
            "blue_preview": pil_to_base64_png(blue_img),
            "hist_r": hist_r.tolist(),
            "hist_g": hist_g.tolist(),
            "hist_b": hist_b.tolist(),
            "mean_rgb": [round(float(arr[:, :, i].mean()), 4) for i in range(3)],
            "std_rgb": [round(float(arr[:, :, i].std()), 4) for i in range(3)],
        },
        "tensor": {
            "shape": [1, 3, 224, 224],
            "min_val": round(float(norm_tensor_chfirst.min()), 3),
            "max_val": round(float(norm_tensor_chfirst.max()), 3),
            "mean_val": round(float(norm_tensor_chfirst.mean()), 3),
            "std_val": round(float(norm_tensor_chfirst.std()), 3),
        },
        "patch": {
            "index": patch_idx,
            "row": row,
            "col": col,
            "size": 16,
            "pixels_count": 16 * 16 * 3,
            "preview": pil_to_base64_png(patch_crop.resize((96, 96), Image.Resampling.NEAREST)),
            "pixels": patch_raw_pixels,
            "mean_rgb": [
                round(float(np.mean(patch_raw_pixels, axis=(0, 1))[i]), 1) for i in range(3)
            ],
        },
    }


@app.post("/api/predict")
async def predict(files: List[UploadFile] = File(...)) -> dict[str, Any]:
    """Exécute une prédiction complète réelle sur ViT et DeiT pour 1 à 20 images."""
    if not files:
        raise HTTPException(status_code=400, detail="Ajoutez au moins une image.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images.")

    vit_bundle = load("vit")
    deit_bundle = load("deit")

    results = []
    for uploaded_file in files:
        image = read_image(await uploaded_file.read())
        vit_res = infer_single(vit_bundle, image)
        deit_res = infer_single(deit_bundle, image)

        agreement = vit_res["predicted_class_id"] == deit_res["predicted_class_id"]
        results.append(
            {
                "filename": uploaded_file.filename or "image.jpg",
                "width": image.width,
                "height": image.height,
                "preview": pil_to_base64_jpeg(image.resize((120, 120))),
                "vit": vit_res,
                "deit": deit_res,
                "agreement": agreement,
            }
        )

    count = len(results)
    agreements = sum(1 for r in results if r["agreement"])
    avg_vit_conf = np.mean([r["vit"]["confidence"] for r in results]) if count else 0
    avg_deit_conf = np.mean([r["deit"]["confidence"] for r in results]) if count else 0

    return {
        "results": results,
        "summary": {
            "total_images": count,
            "agreements": agreements,
            "disagreements": count - agreements,
            "agreement_rate": round(agreements / count, 4) if count else 0,
            "avg_vit_confidence": round(float(avg_vit_conf), 4),
            "avg_deit_confidence": round(float(avg_deit_conf), 4),
        },
        "device": str(DEVICE),
    }


@app.post("/api/attention")
async def get_attention(
    model_name: str = Query("vit", pattern="^(vit|deit)$"),
    layer: int = Query(12, ge=1, le=12),
    head: int = Query(0, ge=0, le=12),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Extrait les matrices d'attention réelles pour une couche et une tête données."""
    image = read_image(await file.read())
    bundle = load(model_name)
    inputs = bundle["processor"](images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = bundle["model"](**inputs, output_attentions=True, return_dict=True)

    attentions = getattr(outputs, "attentions", None)
    if not attentions or len(attentions) < layer or attentions[layer - 1] is None:
        raise HTTPException(
            status_code=500, detail="Impossible d'extraire les attentions de ce modèle."
        )

    raw_layer_heads = attentions[layer - 1][0]  # shape: (12, tokens, tokens)
    if head == 0:
        attn_matrix = raw_layer_heads.mean(dim=0).cpu().numpy()
        heads_averaged = True
    else:
        attn_matrix = raw_layer_heads[head - 1].cpu().numpy()
        heads_averaged = False

    tokens_len = int(attn_matrix.shape[0])
    patch_offset = 2 if (model_name == "deit" and tokens_len >= 198) else 1

    # Attention spatiale du token [CLS] vers les 196 patches
    cls_to_patches = attn_matrix[0, patch_offset : patch_offset + 196]
    cls_min = float(cls_to_patches.min())
    cls_max = float(cls_to_patches.max())
    cls_spread = max(cls_max - cls_min, 1e-8)
    spatial_map = (((cls_to_patches - cls_min) / cls_spread).reshape(14, 14)).tolist()

    dist_spatial_map = []
    if model_name == "deit" and tokens_len >= 198:
        dist_to_patches = attn_matrix[1, patch_offset : patch_offset + 196]
        d_min = float(dist_to_patches.min())
        d_max = float(dist_to_patches.max())
        d_spread = max(d_max - d_min, 1e-8)
        dist_spatial_map = (((dist_to_patches - d_min) / d_spread).reshape(14, 14)).tolist()

    return {
        "model": model_name,
        "layer": layer,
        "head": head,
        "heads_count": 12,
        "tokens": tokens_len,
        "attention": attn_matrix.tolist(),
        "spatial_map": spatial_map,
        "dist_spatial_map": dist_spatial_map,
        "heads_averaged": heads_averaged,
        "cls_attention_sample": [round(float(x), 5) for x in attn_matrix[0, :15]],
    }


@app.post("/api/attention-rollout")
async def get_attention_rollout(
    model_name: str = Query("vit", pattern="^(vit|deit)$"),
    head_fusion: str = Query("mean", pattern="^(mean|max|min)$"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Calcule l'Attention Rollout complète (Abnar & Zuidema, 2020) à travers les 12 couches :
    R^(l) = (0.5 * A^(l) + 0.5 * I) @ R^(l-1)
    """
    image = read_image(await file.read())
    bundle = load(model_name)
    inputs = bundle["processor"](images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = bundle["model"](**inputs, output_attentions=True, return_dict=True)

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise HTTPException(status_code=500, detail="Attentions non disponibles.")

    tokens_len = attentions[0].shape[-1]
    eye = torch.eye(tokens_len, device=DEVICE)
    rollout = torch.eye(tokens_len, device=DEVICE)

    for layer_attn in attentions:
        # layer_attn shape: (1, 12, tokens, tokens)
        if head_fusion == "mean":
            a = layer_attn[0].mean(dim=0)
        elif head_fusion == "max":
            a, _ = layer_attn[0].max(dim=0)
        else:
            a, _ = layer_attn[0].min(dim=0)

        # Ajout de la matrice identité (residual connection) et renormalisation
        a_res = 0.5 * a + 0.5 * eye
        a_norm = a_res / a_res.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(a_norm, rollout)

    rollout_mat = rollout.cpu().numpy()
    patch_offset = 2 if (model_name == "deit" and tokens_len >= 198) else 1
    cls_rollout = rollout_mat[0, patch_offset : patch_offset + 196]

    c_min = float(cls_rollout.min())
    c_max = float(cls_rollout.max())
    c_spread = max(c_max - c_min, 1e-8)
    norm_map = ((cls_rollout - c_min) / c_spread).reshape(14, 14)

    return {
        "model": model_name,
        "rollout_map": norm_map.tolist(),
        "tokens": tokens_len,
        "fusion": head_fusion,
    }


@app.post("/api/tensors/intermediate")
async def get_intermediate_tensors(
    model_name: str = Query("vit", pattern="^(vit|deit)$"),
    patch_idx: int = Query(86, ge=0, le=195),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Extrait les vrais tenseurs et métriques intermédiaires :
    - Filtres de projection appris E (shape 768, 3, 16, 16)
    - Similarité cosinus 2D des encodages positionnels
    - Normes d'activation et ratios résiduels couche par couche (1..12)
    - Sortie des têtes CLS et DIST (pour DeiT)
    """
    image = read_image(await file.read())
    bundle = load(model_name)
    model = bundle["model"]
    processor = bundle["processor"]

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Récupération de l'embedding de patch
    if model_name == "vit":
        embed_module = model.vit.embeddings
    else:
        embed_module = model.deit.embeddings

    patch_conv = embed_module.patch_embeddings.projection  # Conv2d(3, 768, kernel_size=16, stride=16)
    proj_weight = patch_conv.weight.detach().cpu()  # (768, 3, 16, 16)

    # 3 filtres représentatifs
    sample_filters = []
    for f_idx in [0, 42, 128]:
        f_arr = proj_weight[f_idx].numpy()  # (3, 16, 16)
        # Normalisation min-max pour affichage
        f_min, f_max = f_arr.min(), f_arr.max()
        f_norm = ((f_arr - f_min) / max(f_max - f_min, 1e-6) * 255).astype(np.uint8)
        f_img = Image.fromarray(np.transpose(f_norm, (1, 2, 0)))
        sample_filters.append(
            {
                "filter_id": f_idx,
                "preview": pil_to_base64_png(f_img.resize((64, 64), Image.Resampling.NEAREST)),
            }
        )

    # Carte de similarité des position embeddings
    # position_embeddings shape: (1, 197 or 198, 768)
    pos_embed = embed_module.position_embeddings.detach().cpu()[0]
    patch_offset = 2 if (model_name == "deit" and len(pos_embed) >= 198) else 1
    patch_pos = pos_embed[patch_offset : patch_offset + 196]  # (196, 768)

    # Similarité cosinus par rapport au patch sélectionné
    ref_pos = patch_pos[patch_idx].unsqueeze(0)  # (1, 768)
    cos_sim = F.cosine_similarity(patch_pos, ref_pos, dim=-1).numpy().reshape(14, 14)

    # Exécution avec output_hidden_states pour suivre les 12 blocs encodeurs
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states  # tuple de 13 tenseurs (entrée + 12 couches)
    layer_metrics = []

    for l_idx in range(1, len(hidden_states)):
        prev_h = hidden_states[l_idx - 1][0]  # (tokens, 768)
        curr_h = hidden_states[l_idx][0]
        diff = curr_h - prev_h

        curr_norm = float(torch.norm(curr_h, p=2, dim=-1).mean().item())
        diff_norm = float(torch.norm(diff, p=2, dim=-1).mean().item())
        residual_ratio = diff_norm / max(curr_norm, 1e-6)

        layer_metrics.append(
            {
                "layer": l_idx,
                "norm": round(curr_norm, 2),
                "residual_norm": round(diff_norm, 2),
                "residual_ratio": round(residual_ratio, 3),
            }
        )

    # Pour DeiT : analyse distincte des deux têtes CLS et DIST
    deit_dual_head = None
    if model_name == "deit" and hasattr(model, "cls_classifier") and hasattr(model, "distillation_classifier"):
        last_hidden = hidden_states[-1][0]
        cls_token = last_hidden[0]
        dist_token = last_hidden[1]

        cls_logits = model.cls_classifier(cls_token)
        dist_logits = model.distillation_classifier(dist_token)

        cls_probs = torch.softmax(cls_logits, dim=-1)
        dist_probs = torch.softmax(dist_logits, dim=-1)

        cls_top = torch.topk(cls_probs, 3)
        dist_top = torch.topk(dist_probs, 3)

        labels = model.config.id2label

        def fmt_top(val, idx):
            return [
                {
                    "class_id": int(i.item()),
                    "label": str(labels.get(int(i.item()), f"Classe {i.item()}")),
                    "prob": round(float(v.item()), 4),
                }
                for v, i in zip(val, idx)
            ]

        deit_dual_head = {
            "cls_head": fmt_top(cls_top.values, cls_top.indices),
            "dist_head": fmt_top(dist_top.values, dist_top.indices),
            "heads_agree": int(cls_top.indices[0].item()) == int(dist_top.indices[0].item()),
        }

    return {
        "model": model_name,
        "sample_filters": sample_filters,
        "position_similarity_map": cos_sim.tolist(),
        "layer_metrics": layer_metrics,
        "deit_dual_head": deit_dual_head,
    }


@app.post("/api/perturbation")
async def perturbation_experiment(
    perturb_type: str = Query("mask_center", pattern="^(mask_center|mask_random|shuffle)$"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Mode Défi / Perturbation : altère l'image (masquage central, aléatoire ou mélange de patches)
    et compare la prédiction PyTorch originale avec la prédiction perturbée !
    """
    raw = await file.read()
    orig_img = read_image(raw)
    proc_img = preprocess_image(orig_img, mode="resize_crop_256")
    arr = np.array(proc_img).copy()

    # Application de la perturbation
    if perturb_type == "mask_center":
        # Masquer les 4x4 patches centraux (pixels 80..144)
        arr[80:144, 80:144, :] = 128
    elif perturb_type == "mask_random":
        # Masquer 40 patches aléatoires
        np.random.seed(42)
        masked_indices = np.random.choice(196, size=40, replace=False)
        for idx in masked_indices:
            r = idx // 14
            c = idx % 14
            arr[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16, :] = 128
    elif perturb_type == "shuffle":
        # Découper en 196 patches et mélanger
        patches = [
            arr[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16, :].copy()
            for r in range(14)
            for c in range(14)
        ]
        np.random.seed(42)
        np.random.shuffle(patches)
        for idx, patch in enumerate(patches):
            r = idx // 14
            c = idx % 14
            arr[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16, :] = patch

    perturbed_img = Image.fromarray(arr)

    # Inférences
    vit_bundle = load("vit")
    clean_res = infer_single(vit_bundle, proc_img)
    perturbed_res = infer_single(vit_bundle, perturbed_img)

    return {
        "perturb_type": perturb_type,
        "original_preview": pil_to_base64_jpeg(proc_img),
        "perturbed_preview": pil_to_base64_jpeg(perturbed_img),
        "clean_prediction": clean_res,
        "perturbed_prediction": perturbed_res,
        "confidence_drop": round(clean_res["confidence"] - perturbed_res["confidence"], 4),
        "class_changed": clean_res["predicted_class_id"] != perturbed_res["predicted_class_id"],
    }
