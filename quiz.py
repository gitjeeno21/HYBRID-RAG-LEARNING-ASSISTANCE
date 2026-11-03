"""Generate and grade source-grounded interactive study quizzes."""

import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import TOP_K, citation
from hybrid import VectorSearchStore, hybrid_search

AskFunction = Callable[[str], str]

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard", "Mixed"}
VALID_QUESTION_COUNTS = {5, 10, 15}
VALID_QUESTION_TYPES = {"Multiple choice", "Short answer", "True/False", "Mixed"}


def page_number(metadata: Mapping[str, object]) -> int | None:
    label = metadata.get("page_label")
    if isinstance(label, int):
        return label
    page = metadata.get("page")
    return page + 1 if isinstance(page, int) else None


def page_range_filter(chunks: Sequence[Document], page_range: str | None) -> list[Document]:
    """Keep chunks whose source page falls inside an inclusive page range."""
    if not page_range or not page_range.strip():
        return list(chunks)
    match = re.fullmatch(r"\s*(\d+)\s*(?:-|–)\s*(\d+)\s*", page_range)
    if not match:
        raise ValueError("Page range must use the format 'start-end', for example 1-12")
    start, end = (int(value) for value in match.groups())
    if start < 1 or end < start:
        raise ValueError("Page range must be positive and have start <= end")
    return [
        chunk for chunk in chunks
        if (number := page_number(chunk.metadata)) is not None and start <= number <= end
    ]


def retrieve_quiz_chunks(
    vector_store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: Sequence[Document],
    pdf_name: str,
    topic: str,
    difficulty: str = "Medium",
    page_range: str | None = None,
) -> list[Document]:
    """Retrieve quiz evidence with hybrid search restricted to the chosen PDF."""
    n_results = 12 if difficulty == "Hard" else TOP_K
    evidence = hybrid_search(
        topic,
        vector_store,
        bm25,
        list(chunks),
        n_results=n_results,
        metadata_filter={"filename": pdf_name},
    )
    return page_range_filter(evidence, page_range)


QUIZ_PROMPT_TEMPLATE = (
    "Create a practice quiz using ONLY the retrieved text below. Do not use "
    "outside knowledge, assumptions, or facts from memory.\n"
    "The selected PDF is {pdf_name}. The requested topic is: {topic}\n"
    "Difficulty: {difficulty}. Question type: {question_type}. "
    "Generate exactly {question_count} questions.\n\n"
    "Difficulty requirements:\n"
    "- Easy: Bloom's remembering and understanding; recall or definitions supported by one passage. "
    "Assign most questions difficulty \"Easy\".\n"
    "- Medium: Bloom's understanding or applying; explain or apply a concept using one or two passages. "
    "Assign most questions difficulty \"Medium\".\n"
    "- Hard: Bloom's analyzing and evaluating; require why/how reasoning across several passages. "
    "For multiple-choice items, use plausible near-miss distractors grounded in context. "
    "Assign most questions difficulty \"Hard\".\n"
    "- Mixed: approximately 40% Easy, 40% Medium, and 20% Hard.\n\n"
    "Question-type requirements:\n"
    "- Multiple choice: include exactly four distinct options and set answer to the correct option text.\n"
    "- Short answer: use a concise open-answer question; omit options.\n"
    "- True/False: include options [\"True\", \"False\"] and answer \"True\" or \"False\".\n"
    "- Mixed: distribute Multiple choice, Short answer, and True/False across the quiz. "
    "Use the same structure described above for each generated item.\n\n"
    "If the retrieved text does not cover the topic well enough to support "
    "questions and answers, return JSON with covered=false and an empty "
    "questions list. Otherwise return JSON with covered=true and exactly {question_count} "
    "distinct question objects. Each object must have question, answer, difficulty, "
    "type, and source fields; multiple-choice and true/false objects also have options. "
    "Answers must be supported by retrieved text. The source "
    "must exactly match one of the citation labels below. Do not invent "
    "questions, answers, or citations.\n\n"
    "Retrieved text:\n{context}\n\n"
    "Return only valid JSON in this shape:\n"
    '{{"covered": true, "questions": [{{"question": "...", "answer": "...", '
    '"difficulty": "Easy|Medium|Hard", "type": "Multiple choice", '
    '"options": ["...", "...", "...", "..."], "source": "filename, p. X"}}]}}'
)

GRADE_PROMPT_TEMPLATE = (
    "Grade the user's answers to a study quiz using ONLY the retrieved source "
    "context below. The context is the sole authority: do not rely on outside "
    "knowledge or on any assumed answer key. {strictness}\n"
    "For each question, independently "
    "determine the correct answer from that context and compare it to the "
    "user's answer. Use exactly one grade: Correct, Partial, or Incorrect. "
    "Award half credit for Partial and zero for Incorrect. Every correct_answer "
    "must be grounded in the context, and source must exactly match a citation "
    "label present in that context. If the context cannot support an answer, "
    "grade Incorrect and say so in correct_answer.\n\n"
    "Retrieved source context:\n{context}\n\n"
    "Questions and user answers:\n{responses}\n\n"
    "Return only valid JSON with one result per question in the same order:\n"
    '{{"results": [{result_shape}]}}'
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
    difficulty: str = "Medium",
    question_count: int = 5,
    question_type: str = "Mixed",
) -> dict:
    """Generate five structured questions, or report that evidence is absent."""
    if not chunks:
        return {"covered": False, "questions": []}
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    if question_count not in VALID_QUESTION_COUNTS:
        raise ValueError(f"Question count must be one of {sorted(VALID_QUESTION_COUNTS)}")
    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(f"Invalid question type: {question_type}")

    context = build_quiz_context(chunks)
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        pdf_name=pdf_name,
        topic=topic,
        context=context,
        difficulty=difficulty,
        question_count=question_count,
        question_type=question_type,
    )
    response = _call_llm(prompt, ask, "quiz")
    try:
        payload = _parse_json(response)
    except ValueError:
        if ask is None:
            response = _call_llm(
                prompt + "\n\nReturn valid JSON only; do not use Markdown or commentary.",
                None,
                "quiz_retry",
            )
            payload = _parse_json(response)
        else:
            response = ask(prompt + "\n\nReturn valid JSON only; do not use Markdown or commentary.")
            payload = _parse_json(response)
    if payload.get("covered") is False:
        if payload.get("questions") not in ([], None):
            raise ValueError("An uncovered quiz must not contain questions")
        return {"covered": False, "questions": []}

    questions = payload.get("questions")
    if payload.get("covered") is not True or not isinstance(questions, list):
        raise ValueError("Quiz response must include covered=true and questions")
    if len(questions) != question_count:
        raise ValueError(f"Quiz response must contain exactly {question_count} questions, got {len(questions)}")

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
        item_difficulty = item.get("difficulty", difficulty if difficulty != "Mixed" else "Medium")
        item_type = item.get("type", question_type if question_type != "Mixed" else "Short answer")
        options = item.get("options")
        if not all(isinstance(value, str) and value.strip()
                   for value in (question, answer, source)):
            raise ValueError("Each quiz question needs question, answer, and source text")
        if source not in valid_sources:
            raise ValueError(f"Quiz citation does not match retrieved context: {source!r}")
        if item_difficulty not in {"Easy", "Medium", "Hard"}:
            raise ValueError("Each question needs a valid difficulty")
        if not isinstance(item_type, str) or not item_type.strip():
            raise ValueError("Each question needs a valid type")
        if item_type in {"Multiple choice", "True/False"}:
            if not isinstance(options, list) or len(options) != (4 if item_type == "Multiple choice" else 2):
                raise ValueError(f"{item_type} questions need valid options")
            if answer not in options:
                raise ValueError(f"{item_type} answer must match one of its options")
        validated.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "source": source,
                "difficulty": item_difficulty,
                "type": item_type.strip(),
                **({"options": [str(option) for option in options]} if options is not None else {}),
            }
        )
    return {"covered": True, "questions": validated}


def build_grading_prompt(
    questions: Sequence[Mapping[str, str]],
    user_answers: Sequence[str],
    context: str,
) -> str:
    """Build a grading prompt whose only evidence is the retrieved context."""
    if not questions or len(questions) != len(user_answers):
        raise ValueError("Grading requires the same non-empty number of questions and answers")
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
    difficulties = {
        str(question.get("difficulty", "Medium")).lower()
        for question in questions
    }
    if "hard" in difficulties:
        strictness = "For Hard questions, require key terms or correct reasoning; do not accept vague restatement for full credit."
    elif "easy" in difficulties:
        strictness = "For Easy questions, accept partial paraphrases and minor wording differences."
    else:
        strictness = "Apply consistent, evidence-grounded grading to each answer."
    result_shape = ", ".join(
        '{"grade": "Correct|Partial|Incorrect", "feedback": "...", '
        '"correct_answer": "...", "source": "filename, p. X"}'
        for _ in questions
    )
    return GRADE_PROMPT_TEMPLATE.format(
        context=context,
        responses=json.dumps(responses, ensure_ascii=False, indent=2),
        strictness=strictness,
        result_shape=result_shape,
    )


def grade_quiz(
    questions: Sequence[Mapping[str, str]],
    user_answers: Sequence[str],
    context: str,
    ask: AskFunction | None = None,
) -> dict:
    """Grade answers in one source-grounded LLM request."""
    prompt = build_grading_prompt(questions, user_answers, context)
    payload = _parse_json(_call_llm(prompt, ask, "quiz_grade"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(questions):
        raise ValueError(
            f"Grading response must contain exactly {len(questions)} results"
        )

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
