"""划分求解器。

分配单元是连通分量（同患者 + 重复关系），整个分量只能同去训练侧或验证侧。
- 侧别登记表 (registry) 中已有记录的图像，其侧别永久固定，分量必须服从；
- **患者历史侧别**：全部已发布明细中同一患者的侧别同样是硬约束。新图像（图像号从未
  登记）也必须沿用其患者的历史侧别；若历史已把同一患者发布到两侧，则该患者是
  “双侧污染”，任何包含它的划分都不得任选一侧，直接判关联冲突；
- 未锁定的自由分量用多维 0/1 背包 DP 选择一个子集，使验证侧各类张数精确等于配额；
- 无解时给出可解释的原因：关联冲突 / 配额不可达（含最接近状态与剩余分量）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .db import SIDES, TRAIN, VALID
from .parser import Component


@dataclass
class PatientHistory:
    """来自全部已发布清单明细的患者侧别依据。

    sides: 患者号 -> {侧别: [来源清单 id, ...]}，只保存非空侧别。
    manifest_versions: 清单 id -> 输入版本 id，用于在页面上指出来源清单。
    """

    sides: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    manifest_versions: dict[int, int] = field(default_factory=dict)

    def sides_of(self, patient: str) -> dict[str, list[int]]:
        """患者出现过的侧别及来源清单；未发布过的患者返回空字典。"""
        return {
            side: list(ids)
            for side, ids in self.sides.get(patient, {}).items()
            if ids and side in SIDES
        }

    def is_known(self, patient: str) -> bool:
        return bool(self.sides_of(patient))

    def is_conflicted(self, patient: str) -> bool:
        return len(self.sides_of(patient)) > 1

    def side_of(self, patient: str) -> str | None:
        """患者唯一的历史侧别；未发布或已双侧冲突时返回 None。"""
        known = self.sides_of(patient)
        if len(known) == 1:
            return next(iter(known))
        return None

    def sources(self, patient: str) -> list[dict]:
        out: list[dict] = []
        for side, ids in sorted(self.sides_of(patient).items()):
            for mid in ids:
                out.append(
                    {
                        "manifest_id": mid,
                        "version_id": self.manifest_versions.get(mid),
                        "side": side,
                    }
                )
        return out

    def detail(self, patient: str) -> dict[str, list[dict]]:
        return {
            side: [
                {
                    "manifest_id": mid,
                    "version_id": self.manifest_versions.get(mid),
                }
                for mid in ids
            ]
            for side, ids in sorted(self.sides_of(patient).items())
        }

    def _side_label(self, side: str) -> str:
        return "训练侧" if side == TRAIN else "验证侧"

    def describe(self, patient: str) -> str:
        parts = []
        for side, ids in sorted(self.sides_of(patient).items()):
            refs = "、".join(
                f"清单 #{mid}"
                + (
                    f"（v{self.manifest_versions[mid]}）"
                    if mid in self.manifest_versions
                    else ""
                )
                for mid in ids
            )
            parts.append(f"{self._side_label(side)} {refs}")
        return f"患者 {patient} 已发布在" + "；".join(parts)


class SolveError(Exception):
    def __init__(self, stage: str, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.stage = stage          # conflict | quota
        self.reason = reason
        self.detail = detail or {}


@dataclass
class LockedComponent:
    component: Component
    side: str
    locked_images: list[str]
    locked_patients: list[str] = field(default_factory=list)


@dataclass
class SolveResult:
    assignment: dict[int, str]              # cid -> side
    locked: list[LockedComponent]
    free_chosen: list[int]                  # 进入验证侧的自由分量 cid
    target: dict[str, int]
    classes: list[str]


def _side_label(side: str) -> str:
    return "训练侧" if side == TRAIN else "验证侧"


def _vec_add(base: tuple[int, ...], vec: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(base, vec))


def _vec_le(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b))


def solve(
    components: list[Component],
    classes: list[str],
    quota: dict[str, int],
    registry: dict[str, str],
    patient_history: PatientHistory | None = None,
) -> SolveResult:
    target = tuple(quota[c] for c in classes)
    ncls = len(classes)
    zero = tuple(0 for _ in classes)
    history = patient_history or PatientHistory()

    # ---- 1. 处理锁定分量，检测关联冲突 -------------------------------------
    # 锁定依据有三类，任何一类矛盾都判关联冲突：
    #   (a) 图像号永久登记（旧版已发布图像的侧别永不改变）；
    #   (b) 患者历史侧别（已发布明细中该患者的侧别，新图必须沿用）；
    #   (c) 历史双侧污染（同一患者曾被发布到两侧，不得再任选一侧）。
    locked_sum = [0] * ncls
    locked_map: dict[int, LockedComponent] = {}
    for comp in components:
        image_sides = {registry[img] for img in comp.image_nos if img in registry}
        patient_sides: dict[str, set[str]] = {}
        for patient in comp.patients:
            if history.is_known(patient):
                patient_sides[patient] = {
                    side for side, ids in history.sides_of(patient).items() if ids
                }
        polluted = sorted(p for p in comp.patients if history.is_conflicted(p))

        sides = set(image_sides)
        for values in patient_sides.values():
            sides |= values

        if polluted:
            rows = []
            for patient in polluted:
                rows.append(
                    {
                        "patient_no": patient,
                        "sources": history.sources(patient),
                    }
                )
            detail = {
                "component_id": comp.cid,
                "images": comp.image_nos,
                "patients": sorted(comp.patients),
                "conflicted_patients": rows,
                "locked": [
                    {"image_no": img, "side": registry[img]}
                    for img in comp.image_nos
                    if img in registry
                ],
                "patient_history": {
                    patient: history.detail(patient) for patient in polluted
                },
            }
            raise SolveError(
                "conflict",
                "关联冲突："
                + "；".join(history.describe(p) for p in polluted)
                + "。同一患者既在训练侧又在验证侧出现过，历史已被污染；"
                "连通分量必须整体同侧，当前清单不得任选一侧发布。"
                f"（涉及图像：{', '.join(comp.image_nos)}）",
                detail,
            )

        if len(sides) > 1:
            locked_here = [
                (img, registry[img]) for img in comp.image_nos if img in registry
            ]
            detail = {
                "component_id": comp.cid,
                "images": comp.image_nos,
                "patients": sorted(comp.patients),
                "locked": locked_here,
                "image_sides": sorted(image_sides),
                "patient_sides": {
                    patient: sorted(values)
                    for patient, values in sorted(patient_sides.items())
                },
                "patient_history": {
                    patient: history.detail(patient) for patient in sorted(patient_sides)
                },
            }
            image_locks = "、".join(
                f"{img}={_side_label(registry[img])}" for img, _ in locked_here
            )
            patient_locks = "；".join(
                history.describe(patient) for patient in sorted(patient_sides)
            )
            evidence = "；".join(x for x in (image_locks, patient_locks) if x) or "历史依据"
            raise SolveError(
                "conflict",
                f"关联冲突：连通分量 {comp.cid}（含图像 {', '.join(comp.image_nos)}，"
                f"患者 {', '.join(sorted(comp.patients))}）中的历史侧别依据互相矛盾"
                f"（{evidence}），而同患者/重复关系连通的图像必须同侧，无法划分。",
                detail,
            )
        if sides:
            side = sides.pop()
            locked_images = [img for img in comp.image_nos if img in registry]
            locked_patients = sorted(p for p in comp.patients if history.is_known(p))
            locked_map[comp.cid] = LockedComponent(
                comp,
                side,
                locked_images,
                locked_patients=locked_patients,
            )
            if side == VALID:
                for k, cls in enumerate(classes):
                    locked_sum[k] += comp.class_counts.get(cls, 0)

    locked_vec = tuple(locked_sum)
    if not _vec_le(locked_vec, target):
        over = [
            f"{classes[k]} 已固定验证 {locked_vec[k]} 张 > 配额 {target[k]} 张"
            for k in range(ncls)
            if locked_vec[k] > target[k]
        ]
        raise SolveError(
            "quota",
            "配额不可达：曾发布图像的侧别永久固定（含患者历史侧别），已固定在验证侧的图像超出配额——"
            + "；".join(over)
            + "。这些侧别不能更改，请上调对应配额或更正输入。",
            {"locked_vec": locked_vec, "target": target},
        )

    need = tuple(t - l for t, l in zip(target, locked_vec))

    # ---- 2. 自由分量 DP ----------------------------------------------------
    free_comps = [c for c in components if c.cid not in locked_map]
    # 大分量优先，剪枝效果更好
    free_comps.sort(key=lambda c: (-sum(c.class_counts.values()), c.cid))
    vectors: dict[int, tuple[int, ...]] = {
        c.cid: tuple(c.class_counts.get(cls, 0) for cls in classes) for c in free_comps
    }

    chosen: list[int] = []
    if free_comps:
        # 可达状态：当前验证侧增量向量 -> (上一状态, 选入的分量 cid)
        reachable: dict[tuple[int, ...], tuple[tuple[int, ...], int | None]] = {zero: (zero, None)}
        for comp in free_comps:
            v = vectors[comp.cid]
            additions: dict[tuple[int, ...], tuple[tuple[int, ...], int | None]] = {}
            for state in reachable:
                nxt = _vec_add(state, v)
                if _vec_le(nxt, need) and nxt not in reachable and nxt not in additions:
                    additions[nxt] = (state, comp.cid)
            reachable.update(additions)

        if need not in reachable:
            reason, detail = _explain_infeasible(need, reachable, free_comps, vectors, classes, locked_vec)
            raise SolveError("quota", reason, detail)

        # ---- 3. 回溯 -------------------------------------------------------
        state = need
        while state != zero:
            prev, cid = reachable[state]
            assert cid is not None
            chosen.append(cid)
            state = prev
    elif need != zero:
        # 所有分量均已被历史登记锁定，没有可调整的自由分量
        raise SolveError(
            "quota",
            "配额不可达：本版本的全部图像都曾发布且侧别永久固定（或须沿用患者历史侧别），"
            "没有可重新划分的图像，固定在验证侧的计数与新配额不一致。"
            "请恢复原配额，或更正输入引入新图像/新患者。",
            {"locked_vec": locked_vec, "target": target},
        )

    assignment: dict[int, str] = {}
    for cid, lc in locked_map.items():
        assignment[cid] = lc.side
    chosen_set = set(chosen)
    for comp in free_comps:
        assignment[comp.cid] = VALID if comp.cid in chosen_set else TRAIN

    # 最终自检：精确满足配额
    check = [0] * ncls
    for comp in components:
        if assignment[comp.cid] == VALID:
            for k, cls in enumerate(classes):
                check[k] += comp.class_counts.get(cls, 0)
    assert tuple(check) == target, "求解器内部错误：验证侧计数与配额不一致"

    return SolveResult(
        assignment=assignment,
        locked=list(locked_map.values()),
        free_chosen=sorted(chosen),
        target=dict(zip(classes, target)),
        classes=classes,
    )


def _explain_infeasible(
    need: tuple[int, ...],
    reachable: dict[tuple[int, ...], tuple[tuple[int, ...], int | None]],
    free_comps: list[Component],
    vectors: dict[int, tuple[int, ...]],
    classes: list[str],
    locked_vec: tuple[int, ...],
) -> tuple[str, dict]:
    # 最接近目标的可达状态（L1 距离最小，并列取计数较大者）
    def dist(s: tuple[int, ...]) -> tuple[int, int]:
        return (sum(abs(a - b) for a, b in zip(s, need)), -sum(s))

    best = min(reachable.keys(), key=dist)
    best_total = tuple(b + l for b, l in zip(best, locked_vec))
    target_total = tuple(n + l for n, l in zip(need, locked_vec))
    gaps = [
        f"{classes[k]} 需 {target_total[k]} 张，自由分量最多只能凑到 {best_total[k]} 张"
        for k in range(len(classes))
        if best[k] != need[k]
    ]
    # 逐类短缺诊断：某类所有自由分量之和仍不够
    class_totals = [0] * len(classes)
    for comp in free_comps:
        for k, cls in enumerate(classes):
            class_totals[k] += comp.class_counts.get(cls, 0)
    short_classes = [
        f"{classes[k]}：自由分量共含 {class_totals[k]} 张，尚缺 {need[k] - class_totals[k]} 张"
        for k in range(len(classes))
        if class_totals[k] < need[k]
    ]
    detail = {
        "need_from_free": dict(zip(classes, need)),
        "closest_reachable": dict(zip(classes, best_total)),
        "free_components": [
            {
                "component_id": c.cid,
                "images": c.image_nos,
                "patients": sorted(c.patients),
                "vector": {classes[k]: v for k, v in enumerate(vectors[c.cid]) if v},
            }
            for c in free_comps
        ],
    }
    reason = "配额不可达：不存在让验证侧各类张数精确等于配额的分量组合。"
    if short_classes:
        reason += "根本性短缺——" + "；".join(short_classes) + "。"
    if gaps:
        reason += "最接近的可行组合为（" + "；".join(gaps) + "）。"
    reason += (
        "连通分量必须整体同侧，请调整配额，或更正重复关系/患者归属后重试。"
        "注意：已发布图像的侧别永久固定，不参与重新选择。"
    )
    return reason, detail
