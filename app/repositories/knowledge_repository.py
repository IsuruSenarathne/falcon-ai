from typing import List
from sqlalchemy import text
from app.config.database import SessionLocal


class KnowledgeRepository:
    """
    Builds plain-text knowledge documents from the relational data tables
    (courses, advisors, departments, course_modules) for use in the RAG vector store.
    """

    @staticmethod
    def load_documents() -> List[str]:
        db = SessionLocal()
        try:
            rows = db.execute(text("""
                SELECT
                    c.course_code,
                    c.course_name,
                    c.description,
                    c.level,
                    c.credit_hours,
                    c.duration_weeks,
                    c.fee_usd,
                    c.delivery_mode,
                    c.language,
                    c.certification,
                    c.max_enrollment,
                    c.current_enrollment,
                    d.department_name,
                    d.faculty,
                    d.head_of_dept,
                    a.title        AS advisor_title,
                    a.first_name   AS advisor_first,
                    a.last_name    AS advisor_last,
                    a.email        AS advisor_email,
                    a.phone        AS advisor_phone,
                    a.office_location,
                    a.office_hours
                FROM courses c
                LEFT JOIN departments d ON c.department_id = d.department_id
                LEFT JOIN advisors    a ON c.advisor_id    = a.advisor_id
                WHERE c.is_active = 1
            """)).fetchall()

            documents = []
            for r in rows:
                doc = (
                    f"Course: {r.course_name} ({r.course_code}). "
                    f"Department: {r.department_name or 'N/A'}, Faculty: {r.faculty or 'N/A'}. "
                    f"Level: {r.level}. Credits: {r.credit_hours}. "
                    f"Duration: {r.duration_weeks} weeks. Fee: ${r.fee_usd}. "
                    f"Delivery: {r.delivery_mode or 'N/A'}. Language: {r.language or 'English'}. "
                    f"Certification: {r.certification or 'None'}. "
                    f"Enrollment: {r.current_enrollment}/{r.max_enrollment} students. "
                    f"Description: {r.description or 'N/A'}. "
                    f"Advisor: {r.advisor_title or ''} {r.advisor_first or ''} {r.advisor_last or ''}, "
                    f"email: {r.advisor_email or 'N/A'}, "
                    f"phone: {r.advisor_phone or 'N/A'}, "
                    f"office: {r.office_location or 'N/A'}, "
                    f"hours: {r.office_hours or 'N/A'}."
                )
                documents.append(doc)

            # Load course modules as separate documents for finer retrieval
            modules = db.execute(text("""
                SELECT
                    c.course_name,
                    m.module_number,
                    m.module_title,
                    m.description,
                    m.duration_hours,
                    m.is_mandatory
                FROM course_modules m
                JOIN courses c ON m.course_id = c.course_id
                ORDER BY c.course_id, m.module_number
            """)).fetchall()

            for m in modules:
                mandatory = "Mandatory" if m.is_mandatory else "Optional"
                doc = (
                    f"Course: {m.course_name} — Module {m.module_number}: {m.module_title}. "
                    f"{mandatory}. Duration: {m.duration_hours} hours. "
                    f"Description: {m.description or 'N/A'}."
                )
                documents.append(doc)

            return documents
        finally:
            db.close()
