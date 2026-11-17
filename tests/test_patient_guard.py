"""患者侧别保护：新图沿用患者历史侧别、历史双侧污染停供、发布事务按最新依据裁决。"""
import json

import pytest

from app.db import get_db
from app.parser import ImageRow, validate_and_build
from app.service import Service
from app.solver import PatientHistory, SolveError, solve
from app.web import create_app
from tests.conftest import make_csv, make_dup, upload


def _sides(svc):
    m = svc.current_manifest()
    assert m is not None
    return {e["image_no"]: e["side"] for e in svc.get_manifest_entries(m["id"])}


def _history(counts):
    """counts: patient -> [(side, manifest_id), ...]"""
    h = PatientHistory()
    for patient, rows in counts.items():
        for side, mid in rows:
            h.manifest_versions[mid] = 1
            h.sides.setdefault(patient, {}).setdefault(side, []).append(mid)
    return h


# ---------------------------------------------------------------- 求解器单元
def test_new_image_must_follow_patient_history():
    # P1 历史上在训练侧；I9 是 P1 的新图，J1 是全新患者。
    rows = [("I9", "P1", "A"), ("J1", "P2", "A")]
    comps, classes = validate_and_build([ImageRow(*r) for r in rows], [])
    history = _history({"P1": [("train", 1)]})
    res = solve(comps, classes, {"A": 1}, {}, history)
    sides = {img: res.assignment[c.cid] for c in comps for img in c.image_nos}
    assert sides["I9"] == "train"
    assert sides["J1"] == "val"


def test_new_image_patient_history_can_make_quota_infeasible():
    rows = [("I9", "P1", "A")]
    comps, classes = validate_and_build([ImageRow(*r) for r in rows], [])
    history = _history({"P1": [("train", 1)]})
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 1}, {}, history)
    assert ei.value.stage == "quota"


def test_patient_both_sides_is_conflict_even_if_quota_fits():
    rows = [("I9", "P1", "A"), ("J1", "P2", "A")]
    comps, classes = validate_and_build([ImageRow(*r) for r in rows], [])
    history = _history({"P1": [("train", 1), ("val", 2)]})
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 1}, {}, history)
    assert ei.value.stage == "conflict"
    assert "P1" in ei.value.reason
    assert "清单 #1" in ei.value.reason and "清单 #2" in ei.value.reason
    assert ei.value.detail["conflicted_patients"][0]["patient_no"] == "P1"


def test_registry_lock_still_wins_and_patient_history_agrees():
    rows = [("I1", "P1", "A"), ("I2", "P2", "A")]
    comps, classes = validate_and_build([ImageRow(*r) for r in rows], [])
    history = _history({"P1": [("val", 1)]})
    res = solve(comps, classes, {"A": 1}, {"I1": "val"}, history)
    sides = {img: res.assignment[c.cid] for c in comps for img in c.image_nos}
    assert sides["I1"] == "val" and sides["I2"] == "train"


def test_image_lock_contradicting_patient_history_is_conflict():
    rows = [("I1", "P1", "A"), ("I2", "P1", "A")]
    comps, classes = validate_and_build([ImageRow(*r) for r in rows], [])
    history = _history({"P1": [("train", 1)]})
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 1}, {"I1": "val"}, history)
    assert ei.value.stage == "conflict"
    assert "I1=验证侧" in ei.value.reason


# ---------------------------------------------------------------- 服务端流程
def test_delete_old_images_new_image_inherits_patient_side(tmp_path):
    """旧版训练图被删、同患者新图进入更正版：新图必须留在训练侧，验证配额由其他患者补足。"""
    db = str(tmp_path / "g.db")
    svc = Service(db, synchronous=True)
    svc.create_version(
        make_csv([("I1", "P1", "A"), ("I2", "P2", "A")]), make_dup([]), {"A": 0}
    )
    svc.start_partition()
    assert _sides(svc) == {"I1": "train", "I2": "train"}

    # 更正：删去 I1（P1 旧图），新增同患者新图 I9；必须沿用 P1 的训练侧。
    svc.create_version(
        make_csv([("I2", "P2", "A"), ("I9", "P1", "A"), ("J1", "P3", "A")]),
        make_dup([]),
        {"A": 1},
    )
    job = svc.get_job(svc.start_partition())
    assert job["status"] == "succeeded"
    sides = _sides(svc)
    assert sides["I9"] == "train"
    assert sides["J1"] == "val"
    assert sides["I2"] == "train"


def test_no_solution_respecting_patient_history_publishes_nothing(tmp_path):
    db = str(tmp_path / "n.db")
    svc = Service(db, synchronous=True)
    svc.create_version(make_csv([("I1", "P1", "A")]), make_dup([]), {"A": 0})
    svc.start_partition()
    svc.create_version(
        make_csv([("I1", "P1", "A"), ("I9", "P1", "B")]), make_dup([]), {"A": 1, "B": 1}
    )
    job = svc.get_job(svc.start_partition())
    assert job["status"] == "failed"
    assert job["fail_stage"] == "quota"
    assert svc.current_manifest() is None
    conn = get_db(db)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM manifests").fetchone()["c"] == 1
        assert conn.execute("SELECT COUNT(*) c FROM manifest_entries").fetchone()["c"] == 1
    finally:
        conn.close()


def _simulate_legacy_two_sided(tmp_path):
    """模拟旧版本缺陷留下的历史：同一患者 P1 在两个已发布清单中分处两侧。"""
    db = str(tmp_path / "legacy.db")
    svc = Service(db, synchronous=True)
    svc.create_version(
        make_csv([("I1", "P1", "A"), ("I2", "P2", "A")]), make_dup([]), {"A": 0}
    )
    svc.start_partition()  # 当前清单 m1：P1 在训练侧
    m1 = svc.current_manifest()
    conn = get_db(db)
    try:
        cur = conn.execute(
            "INSERT INTO input_versions(image_csv, dup_csv, image_count, class_names, "
            "quotas, status) VALUES (?, '', 2, 'A', '{\"A\": 1}', 'superseded')",
            ("图像号,患者号,类别\nI1,P1,A\nI2,P2,A\n",),
        )
        vid = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO manifests(version_id, status) VALUES (?,'published')", (vid,)
        )
        m_legacy = cur.lastrowid
        conn.execute(
            "INSERT INTO manifest_entries(manifest_id, image_no, patient_no, class_name, "
            "component_id, side) VALUES (?,?,?,?,?,?)",
            (m_legacy, "I1", "P1", "A", 0, "val"),
        )
        conn.commit()
    finally:
        conn.close()
    return svc, m1["id"], m_legacy


def test_legacy_two_sided_patient_blocks_current_manifest_download(tmp_path):
    svc, m1, m_legacy = _simulate_legacy_two_sided(tmp_path)
    leaks = svc.patient_leaks()
    assert [x["patient_no"] for x in leaks] == ["P1"]
    assert {x["manifest_id"] for x in leaks[0]["sources"]} == {m1, m_legacy}
    assert svc.manifest_conflicts()[m1][0]["patient_no"] == "P1"

    app = create_app(svc.db_path)
    c = app.test_client()
    resp = c.get(f"/manifests/{m1}/download/val", follow_redirects=False)
    assert resp.status_code == 302
    page = c.get(f"/manifests/{m1}/download/val", follow_redirects=True).data.decode()
    assert "P1" in page
    assert f"#{m1}" in page and f"#{m_legacy}" in page
    home = c.get("/").data.decode()
    assert "P1" in home and "停供" in home
    hist = c.get("/history").data.decode()
    assert "P1" in hist and "跨" in hist
    app.extensions["service"].close()
    svc.close()


def test_legacy_two_sided_patient_makes_later_partition_conflict(tmp_path):
    svc, _, _ = _simulate_legacy_two_sided(tmp_path)
    # 更正后当前版本仍包含 P1 的图像：不得任选一侧，求解阶段即判冲突。
    svc.create_version(
        make_csv([("I1", "P1", "A"), ("I2", "P2", "A"), ("J1", "P3", "A")]),
        make_dup([]),
        {"A": 1},
    )
    job = svc.get_job(svc.start_partition())
    assert job["status"] == "failed"
    assert job["fail_stage"] == "conflict"
    assert "P1" in job["reason"]
    detail = json.loads(job["detail"])
    assert detail["conflicted_patients"][0]["patient_no"] == "P1"
    assert svc.current_manifest() is None
    svc.close()


def test_publish_rechecks_patient_basis_and_leaves_no_half_manifest(tmp_path):
    """求解后、发布前新增了与患者历史侧别矛盾的已发布依据：发布必须整体失败。"""
    db = str(tmp_path / "race.db")
    svc = Service(db, synchronous=True)
    svc.create_version(make_csv([("I1", "P1", "A")]), make_dup([]), {"A": 0})
    svc.start_partition()
    # 新版本 + 待发布的手工数据（I9 与 J1 都放 val，违反 P1 历史 train）
    vid, _ = svc.create_version(
        make_csv([("I9", "P1", "A"), ("J1", "P2", "A")]), make_dup([]), {"A": 1}
    )
    conn = get_db(db)
    try:
        job_id = conn.execute(
            "INSERT INTO partition_jobs(version_id,status) VALUES (?,'running')", (vid,)
        ).lastrowid
        conn.commit()
    finally:
        conn.close()
    entries = [
        {"image_no": "I9", "patient_no": "P1", "class_name": "A",
         "component_id": 0, "side": "val"},
        {"image_no": "J1", "patient_no": "P2", "class_name": "A",
         "component_id": 1, "side": "train"},
    ]
    svc._publish(job_id, vid, {"entries": entries, "classes": ["A"],
                               "quota": {"A": 1}, "locked": []})
    job = svc.get_job(job_id)
    assert job["status"] == "failed"
    assert job["fail_stage"] == "conflict"
    conn = get_db(db)
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM manifests WHERE version_id=?", (vid,)
        ).fetchone()["c"] == 0
        assert conn.execute(
            "SELECT COUNT(*) c FROM manifest_entries WHERE image_no='I9'"
        ).fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM side_registry").fetchone()["c"] == 1
    finally:
        conn.close()
    svc.close()


def test_web_correction_new_image_inherits_patient_side(client):
    c, svc = client
    upload(c, [("I1", "P1", "A"), ("I2", "P2", "A")], {"A": 0})
    c.post("/partition", follow_redirects=True)
    upload(c, [("I2", "P2", "A"), ("I9", "P1", "A"), ("J1", "P3", "A")], {"A": 1})
    c.post("/partition", follow_redirects=True)
    sides = _sides(svc)
    assert sides["I9"] == "train"
    home = c.get("/").data.decode()
    assert "泄漏" not in home  # 没有双侧污染时不误报
