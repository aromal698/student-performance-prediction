"""Shared Supabase database layer for EduPredict SPP.

Both Streamlit portals use this module so Student/Tutor and Principal read/write
one shared database instead of each app's local filesystem.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import streamlit as st
from supabase import create_client, Client


def _setting(name: str) -> str:
    # Streamlit Cloud secrets are case-sensitive. Support the recommended
    # SUPABASE_KEY and common SECRET_KEY naming so deployment is less fragile.
    aliases = {
        "SUPABASE_KEY": ["SUPABASE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"],
        "SUPABASE_URL": ["SUPABASE_URL", "SUPABASE_PROJECT_URL"],
    }
    value = ""
    for candidate in aliases.get(name, [name]):
        try:
            value = st.secrets.get(candidate, "")
        except Exception:
            value = ""
        if value:
            break
    if value is None:
        return ""
    return str(value).strip()


def get_supabase() -> Client:
    url = _setting("SUPABASE_URL").rstrip("/")
    key = _setting("SUPABASE_KEY")
    if url.endswith("/rest/v1"):
        url = url[:-8]
    if not url or not key:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to this Streamlit app's Secrets."
        )
    return create_client(url, key)


def _rows(response):
    return list(getattr(response, "data", None) or [])


def test_connection() -> tuple[bool, str]:
    try:
        sb = get_supabase()
        sb.table("students").select("university_id").limit(1).execute()
        return True, "Connected to Supabase successfully."
    except Exception as exc:
        return False, f"Supabase connection failed: {exc}"


def get_students():
    return _rows(get_supabase().table("students").select("*").order("registered_at", desc=True).execute())


def get_student(university_id: str):
    rows = _rows(
        get_supabase().table("students").select("*")
        .eq("university_id", str(university_id).strip()).limit(1).execute()
    )
    return rows[0] if rows else None


def save_student(university_id, student_name, department, semester, studied_college, registered_by, active=True):
    row = {
        "university_id": str(university_id).strip(),
        "student_name": str(student_name).strip(),
        "department": str(department).strip(),
        "semester": str(semester).strip().upper(),
        "studied_college": str(studied_college or "").strip(),
        "registered_by": str(registered_by or "").strip(),
        "active": bool(active),
    }
    return get_supabase().table("students").upsert(row, on_conflict="university_id").execute()


def delete_student(university_id: str):
    uid = str(university_id).strip()
    return get_supabase().table("students").delete().eq("university_id", uid).execute()


def get_tutors():
    return _rows(get_supabase().table("tutors").select("*").order("registered_at", desc=True).execute())


def save_tutor(tutor_id, tutor_name, department, semester, credit_score=0, active=True):
    row = {
        "tutor_id": str(tutor_id).strip(),
        "tutor_name": str(tutor_name).strip(),
        "department": str(department).strip(),
        "semester": str(semester or "").strip().upper(),
        "credit_score": float(credit_score or 0),
        "active": bool(active),
    }
    return get_supabase().table("tutors").upsert(row, on_conflict="tutor_id").execute()


def get_marks(university_id=None, department=None, semester=None):
    query = get_supabase().table("student_marks").select("*")
    if university_id:
        query = query.eq("university_id", str(university_id).strip())
    if department:
        query = query.eq("department", str(department).strip())
    if semester:
        query = query.eq("semester", str(semester).strip().upper())
    return _rows(query.order("submitted_at", desc=True).execute())


def save_marks(university_id, department, semester, tutor_name, subjects):
    # One active mark submission per student. Re-submitting replaces the current
    # record, which makes the student portal deterministic and easy to monitor.
    sb = get_supabase()
    uid = str(university_id).strip()
    sb.table("student_marks").delete().eq("university_id", uid).execute()
    row = {
        "university_id": uid,
        "department": str(department).strip(),
        "semester": str(semester).strip().upper(),
        "tutor_name": str(tutor_name or "").strip(),
        "subjects": subjects if isinstance(subjects, list) else json.loads(subjects),
    }
    return sb.table("student_marks").insert(row).execute()


def delete_marks(university_id: str):
    return get_supabase().table("student_marks").delete().eq("university_id", str(university_id).strip()).execute()


def get_smart_cards():
    return _rows(get_supabase().table("smart_cards").select("*").order("submitted_at", desc=True).execute())


def save_smart_card(row: dict):
    cgpa = row.get("CGPA", "")
    try:
        cgpa = float(cgpa) if str(cgpa).strip() else None
    except Exception:
        cgpa = None
    uid = str(row.get("University_ID", "")).strip()
    # The SQL schema links this field to students. Keep it NULL when the optional
    # ID is not yet a tutor-registered student, so Smart Card creation never fails.
    if uid:
        try:
            if not get_student(uid):
                uid = ""
        except Exception:
            uid = ""
    payload = {
        "registration_id": str(row.get("Registration_ID", "")).strip(),
        "university_id": uid or None,
        "name": str(row.get("Student_Name", row.get("Name", ""))).strip(),
        "dob": str(row.get("DOB", "")).strip() or None,
        "blood_group": str(row.get("Blood_Group", "")).strip(),
        "address": str(row.get("Address", "")).strip(),
        "pin_code": str(row.get("PIN_Code", "")).strip(),
        "studied_college": str(row.get("Studied_College", "")).strip(),
        "department": str(row.get("Department", "")).strip(),
        "semester": str(row.get("Semester", "")).strip().upper(),
        "cgpa": cgpa,
        "university_name": str(row.get("University_Name", "")).strip(),
    }
    return get_supabase().table("smart_cards").upsert(payload, on_conflict="registration_id").execute()


def get_files(university_id=None):
    query = get_supabase().table("student_files").select("*")
    if university_id:
        query = query.eq("university_id", str(university_id).strip())
    return _rows(query.order("uploaded_at", desc=True).execute())


def save_file_metadata(university_id, file_name, file_path="", uploaded_by=""):
    return get_supabase().table("student_files").insert({
        "university_id": str(university_id).strip(),
        "file_name": str(file_name).strip(),
        "file_path": str(file_path or ""),
        "uploaded_by": str(uploaded_by or ""),
    }).execute()


def add_audit(role, username, action, university_id="", department="", semester="", details=""):
    return get_supabase().table("audit_logs").insert({
        "username": str(username or "").strip(),
        "role": str(role or "").strip(),
        "action": str(action or "").strip(),
        "university_id": str(university_id or "").strip(),
        "department": str(department or "").strip(),
        "semester": str(semester or "").strip().upper(),
        "details": str(details or "").strip(),
    }).execute()


def get_audit_logs():
    return _rows(get_supabase().table("audit_logs").select("*").order("timestamp", desc=True).execute())
