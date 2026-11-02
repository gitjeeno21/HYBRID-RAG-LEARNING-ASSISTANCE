"""Generate and grade source-grounded interactive study quizzes."""

import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import citation
from hybrid import VectorSearchStore, hybrid_search

AskFunction = Callable[[str], str]


def retrieve_quiz_chunks(
    vector_store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: Sequence[Document],
    pdf_name: str,
    topic: str,
) -> list[Document]:
    """Retrieve quiz evidence with hybrid search restricted to the chosen PDF."""
    return hybrid_search(
        topic,
        vector_store,
        bm25,
        list(chunks),
        metadata_filter={"filename": pdf_name},
    )


QUIZ_PROMPT_TEMPLATE = (
    "Create a practice quiz using ONLY the retrieved text below. Do not use "
    "outside knowledge, assumptions, or facts from memory.\n"
    "The selected PDF is {pdf_name}. The requested topic is: {topic}\n\n"
    "If the retrieved text does not cover the topic well enough to support "
    "questions and answers, return JSON with covered=false and an empty "
    "questions list. Otherwise return JSON with covered=true and exactly five "
    "distinct question objects. Each object must have question, answer, and "
    "source fields. Answers must be supported by retrieved text. The source "
    "must exactly match one of the citation labels below. Do not invent "
    "questions, answers, or citations.\n\n"
    "Retrieved text:\n{context}\n\n"
    "Return only valid JSON in this shape:\n"
    '{{"covered": true, "questions": [{{"question": "...", '
    '"answer": "...", "source": "filename, p. X"}}, '
    '{{"question": "...", "answer": "...", "source": "filename, p. X"}}, '
    '{{"question": "...", "answer": "...", "source": "filename, p. X"}}, '
    '{{"question": "...", "answer": "...", "source": "filename, p. X"}}, '
    '{{"question": "...", "answer": "...", "source": "filename, p. X"}}]}}'
)

GRADE_PROMPT_TEMPLATE = (
    "Grade the user's answers to a study quiz using ONLY the retrieved source "
    "context below. The context is the sole authority: do not rely on outside "
    "knowledge or on any assumed answer key. For each question, independently "
    "determine the correct answer from that context and compare it to the "
    "user's answer. Use exactly one grade: Correct, Partial, or Incorrect. "
    "Award half credit for Partial and zero for Incorrect. Every correct_answer "
    "must be grounded in the context, and source must exactly match a citation "
    "label present in that context. If the context cannot support an answer, "
    "grade Incorrect and say so in correct_answer.\n\n"
    "Retrieved source context:\n{context}\n\n"
    "Questions and user answers:\n{responses}\n\n"
    "Return only valid JSON with one result per question in the same order:\n"
    '{{"results": [{{"grade": "Correct|Partial|Incorrect", '
    '"feedback": "...", "correct_answer": "...", '
    '"source": "filename, p. X"}}, '
    '{{"grade": "Correct|Partial|Incorrect", "feedback": "...", '
    '"correct_answer": "...", "source": "filename, p. X"}}, '
    '{{"grade": "Correct|Partial|Incorrect", "feedback": "...", '
    '"correct_answer": "...", "source": "filename, p. X"}}, '
    '{{"grade": "Correct|Partial|Incorrect", "feedback": "...", '
    '"correct_answer": "...", "source": "filename, p. X"}}, '
    '{{"grade": "Correct|Partial|Incorrect", "feedback": "...", '
    '"correct_answer": "...", "source": "filename, p. X"}}]}}'
)

VALID_GRADES = {"Correct", "Partial", "Incorrect"}


def build_quiz_context(chunks: Sequence[Document]) -> str:
    """Format the exact cited passages used for quiz generation and grading."""
    return "\n\n".join(
        f"Source: {citation(chunk.metadata)}\n{chunk.page_content}"
        for chunk in chunks
    )


def _call_llm(prompt: str, ask: AskFunction | None, label: str) -> str:
    """Use an injected mock or the project's retrying Groq helper."""
    if ask is not None:
        return ask(prompt)
    from evaluator import call_llm

    return call_llm(prompt, temperature=0.2, label=label)


def _parse_json(reply: str) -> dict:
    """Parse model JSON while tolerating a surrounding Markdown code fence."""
    cleaned = reply.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Quiz model returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Quiz model response must be a JSON object")
    return parsed


def generate_quiz(
    pdf_name: str,
    topic: str,
    chunks: Sequence[Document],
    ask: AskFunction | None = None,
) -> dict:
    """Generate five structured questions, or report that evidence is absent."""
    if not chunks:
        return {"covered": False, "questions": []}

    context = build_quiz_context(chunks)
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        pdf_name=pdf_name,
        topic=topic,
        context=context,
    )
    response = _call_llm(prompt, ask, "quiz")
    payload = _parse_json(response)
    if payload.get("covered") is False:
        if payload.get("questions") not in ([], None):
            raise ValueError("An uncovered quiz must not contain questions")
        return {"covered": False, "questions": []}

    questions = payload.get("questions")
    if payload.get("covered") is not True or not isinstance(questions, list):
        raise ValueError("Quiz response must include covered=true and questions")
    if len(questions) != 5:
        raise ValueError(f"Quiz response must contain exactly five questions, got {len(questions)}")

    valid_sources = {
        citation(chunk.metadata)
        for chunk in chunks
    }
    validated = []
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each quiz question must be a JSON object")
        question = item.get("question")
        answer = item.get("answer")
        source = item.get("source")
        if not all(isinstance(value, str) and value.strip()
                   for value in (question, answer, source)):
            raise ValueError("Each quiz question needs question, answer, and source text")
        if source not in valid_sources:
            raise ValueError(f"Quiz citation does not match retrieved context: {source!r}")
        validated.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "source": source,
            }
        )
    return {"covered": True, "questions": validated}


def build_grading_prompt(
    questions: Sequence[Mapping[str, str]],
    user_answers: Sequence[str],
    context: str,
) -> str:
    """Build a grading prompt whose only evidence is the retrieved context."""
    if len(questions) != 5 or len(user_answers) != 5:
        raise ValueError("Grading requires five questions and five user answers")
    responses = [
        {
            "number": index,
            "question": question["question"],
            "user_answer": user_answer,
        }
        for index, (question, user_answer) in enumerate(
            zip(questions, user_answers), start=1
        )
    ]
    return GRADE_PROMPT_TEMPLATE.format(
        context=context,
        responses=json.dumps(responses, ensure_ascii=False, indent=2),
    )


def grade_quiz(
    questions: Sequence[Mapping[str, str]],
    user_answers: Sequence[str],
    context: str,
    ask: AskFunction | None = None,
) -> dict:
    """Grade all five answers in one source-grounded LLM request."""
    prompt = build_grading_prompt(questions, user_answers, context)
    payload = _parse_json(_call_llm(prompt, ask, "quiz_grade"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 5:
        raise ValueError("Grading response must contain exactly five results")

    valid_sources = {
        source.strip()
        for source in re.findall(r"^Source:\s*(.+)$", context, flags=re.M)
    }
    validated_results = []
    points = 0.0
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Each grading result must be a JSON object")
        grade = item.get("grade")
        feedback = item.get("feedback")
        correct_answer = item.get("correct_answer")
        source = item.get("source")
        if grade not in VALID_GRADES:
            raise ValueError(f"Invalid quiz grade: {grade!r}")
        if not all(isinstance(value, str) and value.strip()
                   for value in (feedback, correct_answer, source)):
            raise ValueError("Each grading result needs feedback, correct answer, and source")
        if source not in valid_sources:
            raise ValueError(f"Grading citation does not match retrieved context: {source!r}")
        points += {"Correct": 1.0, "Partial": 0.5, "Incorrect": 0.0}[grade]
        validated_results.append(
            {
                "grade": grade,
                "feedback": feedback.strip(),
                "correct_answer": correct_answer.strip(),
                "source": source,
            }
        )
    return {
        "results": validated_results,
        "score": points,
        "maximum": len(validated_results),
    }


def make_quiz_state(
    pdf_name: str,
    topic: str,
    quiz: Mapping[str, object],
    chunks: Sequence[Document],
) -> dict:
    """Create the serializable session state for an active quiz attempt."""
    questions = quiz.get("questions")
    if quiz.get("covered") is not True or not isinstance(questions, list):
        raise ValueError("Cannot start a quiz without five grounded questions")
    return {
        "attempt_id": uuid.uuid4().hex,
        "pdf": pdf_name,
        "topic": topic,
        "questions": questions,
        "answers": [""] * len(questions),
        "grade": None,
        "context": build_quiz_context(chunks),
        "chunks": [
            {"citation": citation(chunk.metadata), "text": chunk.page_content}
            for chunk in chunks
        ],
    }


def submit_quiz_state(
    state: Mapping[str, object],
    user_answers: Sequence[str],
    ask: AskFunction | None = None,
) -> dict:
    """Advance active quiz state to graded state without discarding answers."""
    questions = state["questions"]
    context = state["context"]
    if not isinstance(questions, list) or not isinstance(context, str):
        raise ValueError("Quiz state is missing questions or retrieved context")
    grade = grade_quiz(questions, user_answers, context, ask=ask)
    return {
        **state,
        "answers": list(user_answers),
        "grade": grade,
    }
