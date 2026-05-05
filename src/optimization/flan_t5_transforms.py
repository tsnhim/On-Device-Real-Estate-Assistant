from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.utils.prune as prune
from transformers.models.t5.modeling_t5 import T5Attention


def _linear_family(name: str) -> str:
    if "SelfAttention" in name or "EncDecAttention" in name:
        return "attention_linear"
    if "DenseReluDense" in name:
        return "mlp_linear"
    return "other_linear"


def iter_named_linear_modules(model: torch.nn.Module, family: str = "all_linear") -> list[tuple[str, torch.nn.Linear]]:
    matches: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_family = _linear_family(name)
            if family == "all_linear" or family == module_family:
                matches.append((name, module))
    return matches


def iter_named_attention_modules(model: torch.nn.Module) -> list[tuple[str, T5Attention]]:
    matches: list[tuple[str, T5Attention]] = []
    for name, module in model.named_modules():
        if isinstance(module, T5Attention):
            matches.append((name, module))
    return matches


def _replace_submodule(root: torch.nn.Module, module_path: str, replacement: torch.nn.Module) -> None:
    if "." in module_path:
        parent_path, attr_name = module_path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        attr_name = module_path
    setattr(parent, attr_name, replacement)


def _new_linear_like(module: torch.nn.Linear, *, out_features: int, in_features: int) -> torch.nn.Linear:
    replacement = torch.nn.Linear(in_features, out_features, bias=module.bias is not None)
    replacement = replacement.to(dtype=module.weight.dtype, device=module.weight.device)
    if module.bias is None:
        replacement.bias = None
    return replacement


def quantify_runtime_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype cast: {dtype_name}")
    return mapping[dtype_name]


def cast_model_dtype(model: torch.nn.Module, dtype_name: str) -> torch.nn.Module:
    return model.to(dtype=quantify_runtime_dtype(dtype_name))


def apply_unstructured_pruning(
    model: torch.nn.Module,
    *,
    family: str,
    amount: float,
) -> dict[str, Any]:
    parameters_to_prune = [(module, "weight") for _, module in iter_named_linear_modules(model, family=family)]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    return build_unstructured_pruning_audit(model, family=family)


def build_unstructured_pruning_audit(model: torch.nn.Module, family: str) -> dict[str, Any]:
    zero_count = 0
    total_count = 0
    layers = []
    for name, module in iter_named_linear_modules(model, family=family):
        layer_zero = int((module.weight.detach() == 0).sum().item())
        layer_total = int(module.weight.numel())
        zero_count += layer_zero
        total_count += layer_total
        layers.append(
            {
                "name": name,
                "family": _linear_family(name),
                "zero_weights": layer_zero,
                "total_weights": layer_total,
                "sparsity": round(layer_zero / layer_total, 6) if layer_total else 0.0,
            }
        )
    return {
        "family": family,
        "linear_layer_count": len(layers),
        "zero_weights": zero_count,
        "total_weights": total_count,
        "global_linear_sparsity": round(zero_count / total_count, 6) if total_count else 0.0,
        "layers": layers,
    }


def prune_structured_mlp_intermediate(
    model: torch.nn.Module,
    *,
    amount: float,
) -> dict[str, Any]:
    entries = []
    layer_summaries = []
    for name, module in model.named_modules():
        if hasattr(module, "wi_0") and hasattr(module, "wi_1") and hasattr(module, "wo"):
            wi_0 = module.wi_0
            wi_1 = module.wi_1
            wo = module.wo
            if not (isinstance(wi_0, torch.nn.Linear) and isinstance(wi_1, torch.nn.Linear) and isinstance(wo, torch.nn.Linear)):
                continue
            keep_count = max(1, int(round(wi_0.out_features * (1.0 - amount))))
            wi_0_scores = wi_0.weight.detach().abs().sum(dim=1)
            wi_1_scores = wi_1.weight.detach().abs().sum(dim=1)
            wo_scores = wo.weight.detach().abs().sum(dim=0)
            scores = wi_0_scores + wi_1_scores + wo_scores
            keep_indices = torch.topk(scores, k=keep_count, largest=True).indices.sort().values.tolist()
            apply_structured_mlp_entry(model, {"module_path": name, "keep_indices": keep_indices})
            entries.append({"module_path": name, "keep_indices": keep_indices})
            layer_summaries.append(
                {
                    "module_path": name,
                    "original_size": int(wi_0.out_features),
                    "kept_size": int(keep_count),
                    "sparsity": round(1.0 - (keep_count / wi_0.out_features), 6),
                }
            )
    return {
        "operation": "prune_structured_mlp",
        "amount": amount,
        "entries": entries,
        "summary": {
            "layer_count": len(layer_summaries),
            "layers": layer_summaries,
        },
    }


def apply_structured_mlp_entry(model: torch.nn.Module, entry: dict[str, Any]) -> None:
    module = model.get_submodule(entry["module_path"])
    keep = torch.tensor(entry["keep_indices"], dtype=torch.long, device=module.wi_0.weight.device)

    wi_0_old = module.wi_0
    wi_1_old = module.wi_1
    wo_old = module.wo

    wi_0_new = _new_linear_like(wi_0_old, out_features=len(keep), in_features=wi_0_old.in_features)
    wi_1_new = _new_linear_like(wi_1_old, out_features=len(keep), in_features=wi_1_old.in_features)
    wo_new = _new_linear_like(wo_old, out_features=wo_old.out_features, in_features=len(keep))

    wi_0_new.weight.data.copy_(wi_0_old.weight.data.index_select(0, keep))
    wi_1_new.weight.data.copy_(wi_1_old.weight.data.index_select(0, keep))
    wo_new.weight.data.copy_(wo_old.weight.data.index_select(1, keep))
    if wi_0_old.bias is not None:
        wi_0_new.bias.data.copy_(wi_0_old.bias.data.index_select(0, keep))
    if wi_1_old.bias is not None:
        wi_1_new.bias.data.copy_(wi_1_old.bias.data.index_select(0, keep))
    if wo_old.bias is not None:
        wo_new.bias.data.copy_(wo_old.bias.data)

    module.wi_0 = wi_0_new
    module.wi_1 = wi_1_new
    module.wo = wo_new


def prune_structured_attention_heads(
    model: torch.nn.Module,
    *,
    heads_to_prune: int,
) -> dict[str, Any]:
    entries = []
    layer_summaries = []
    for name, module in iter_named_attention_modules(model):
        original_heads = int(module.n_heads)
        if original_heads <= heads_to_prune or heads_to_prune <= 0:
            continue
        head_dim = int(module.key_value_proj_dim)
        keep_count = original_heads - heads_to_prune
        scores = _attention_head_scores(module)
        keep_heads = torch.topk(scores, k=keep_count, largest=True).indices.sort().values.tolist()
        entry = {
            "module_path": name,
            "keep_head_indices": keep_heads,
            "head_dim": head_dim,
            "original_head_count": original_heads,
            "new_head_count": keep_count,
            "has_relative_attention_bias": bool(hasattr(module, "relative_attention_bias")),
        }
        apply_structured_attention_entry(model, entry)
        entries.append(entry)
        layer_summaries.append(
            {
                "module_path": name,
                "original_head_count": original_heads,
                "kept_head_count": keep_count,
                "sparsity": round(1.0 - (keep_count / original_heads), 6),
            }
        )
    return {
        "operation": "prune_structured_attention",
        "heads_to_prune": heads_to_prune,
        "entries": entries,
        "summary": {
            "layer_count": len(layer_summaries),
            "layers": layer_summaries,
        },
    }


def _attention_head_scores(module: T5Attention) -> torch.Tensor:
    head_dim = int(module.key_value_proj_dim)
    q_scores = module.q.weight.detach().abs().view(module.n_heads, head_dim, module.q.in_features).sum(dim=(1, 2))
    k_scores = module.k.weight.detach().abs().view(module.n_heads, head_dim, module.k.in_features).sum(dim=(1, 2))
    v_scores = module.v.weight.detach().abs().view(module.n_heads, head_dim, module.v.in_features).sum(dim=(1, 2))
    o_scores = module.o.weight.detach().abs().view(module.o.out_features, module.n_heads, head_dim).sum(dim=(0, 2))
    score = q_scores + k_scores + v_scores + o_scores
    if hasattr(module, "relative_attention_bias"):
        score = score + module.relative_attention_bias.weight.detach().abs().sum(dim=0)
    return score


def apply_structured_attention_entry(model: torch.nn.Module, entry: dict[str, Any]) -> None:
    module = model.get_submodule(entry["module_path"])
    keep_heads = entry["keep_head_indices"]
    head_dim = int(entry["head_dim"])
    row_keep = []
    for head_idx in keep_heads:
        row_keep.extend(range(head_idx * head_dim, (head_idx + 1) * head_dim))
    keep = torch.tensor(row_keep, dtype=torch.long, device=module.q.weight.device)

    q_old = module.q
    k_old = module.k
    v_old = module.v
    o_old = module.o

    q_new = _new_linear_like(q_old, out_features=len(keep), in_features=q_old.in_features)
    k_new = _new_linear_like(k_old, out_features=len(keep), in_features=k_old.in_features)
    v_new = _new_linear_like(v_old, out_features=len(keep), in_features=v_old.in_features)
    o_new = _new_linear_like(o_old, out_features=o_old.out_features, in_features=len(keep))

    q_new.weight.data.copy_(q_old.weight.data.index_select(0, keep))
    k_new.weight.data.copy_(k_old.weight.data.index_select(0, keep))
    v_new.weight.data.copy_(v_old.weight.data.index_select(0, keep))
    o_new.weight.data.copy_(o_old.weight.data.index_select(1, keep))
    if q_old.bias is not None:
        q_new.bias.data.copy_(q_old.bias.data.index_select(0, keep))
    if k_old.bias is not None:
        k_new.bias.data.copy_(k_old.bias.data.index_select(0, keep))
    if v_old.bias is not None:
        v_new.bias.data.copy_(v_old.bias.data.index_select(0, keep))
    if o_old.bias is not None:
        o_new.bias.data.copy_(o_old.bias.data)

    module.q = q_new
    module.k = k_new
    module.v = v_new
    module.o = o_new
    module.n_heads = int(entry["new_head_count"])
    module.inner_dim = module.n_heads * head_dim
    if hasattr(module, "relative_attention_bias"):
        old_bias = module.relative_attention_bias
        new_bias = torch.nn.Embedding(old_bias.num_embeddings, module.n_heads)
        new_bias = new_bias.to(dtype=old_bias.weight.dtype, device=old_bias.weight.device)
        head_keep = torch.tensor(entry["keep_head_indices"], dtype=torch.long, device=old_bias.weight.device)
        new_bias.weight.data.copy_(old_bias.weight.data.index_select(1, head_keep))
        module.relative_attention_bias = new_bias


def quantize_dynamic_linear_layers(model: torch.nn.Module, dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)


def apply_pipeline_step_for_artifact(
    model: torch.nn.Module,
    step: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    op = step["op"]
    if op == "cast_dtype":
        model = cast_model_dtype(model, step["dtype"])
        return model, {"operation": op, "dtype": step["dtype"]}
    if op == "dynamic_quantize_linear":
        model = quantize_dynamic_linear_layers(model, dtype=torch.qint8)
        return model, {"operation": op, "dtype": step["dtype"], "module_types": ["torch.nn.Linear"]}
    if op == "prune_unstructured":
        audit = apply_unstructured_pruning(model, family=step["family"], amount=step["amount"])
        return model, {
            "operation": op,
            "criterion": step["criterion"],
            "family": step["family"],
            "amount": step["amount"],
            "audit": audit,
        }
    if op == "prune_structured_mlp":
        audit = prune_structured_mlp_intermediate(model, amount=step["amount"])
        return model, audit
    if op == "prune_structured_attention":
        audit = prune_structured_attention_heads(model, heads_to_prune=step["heads_to_prune"])
        return model, audit
    raise ValueError(f"Unsupported artifact pipeline op: {op}")


def apply_pipeline_step_for_loading(model: torch.nn.Module, step: dict[str, Any]) -> torch.nn.Module:
    op = step["op"]
    if op == "cast_dtype":
        return cast_model_dtype(model, step["dtype"])
    if op == "dynamic_quantize_linear":
        return quantize_dynamic_linear_layers(model, dtype=torch.qint8)
    if op == "prune_structured_mlp":
        for entry in step["entries"]:
            apply_structured_mlp_entry(model, entry)
        return model
    if op == "prune_structured_attention":
        for entry in step["entries"]:
            apply_structured_attention_entry(model, entry)
        return model
    if op == "prune_unstructured":
        return model
    raise ValueError(f"Unsupported loader pipeline op: {op}")

