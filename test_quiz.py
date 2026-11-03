"""Offline tests for grounded quiz generation, submission, and grading."""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from langchain_core.documents import Document

from quiz import (
    build_grading_prompt,
    build_quiz_context,
    generate_quiz,
    make_quiz_state,
    submit_quiz_state,
)


class QuizGradingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        from main import app
        self.client = TestClient(app)
        # The payloads cite ML222.pdf; make the library contain it so these
        # tests do not depend on which PDFs happen to be in data/.
        pdfs = patch("main.list_pdf_names", return_value=["ML222.pdf"])
        pdfs.start()
        self.addCleanup(pdfs.stop)

    def _payload(self, answers, question_type="Multiple choice", count=5):
        questions = []
        for index in range(count):
            question = {
                "question": f"Question {index + 1}",
                "answer": "Correct",
                "source": "ML222.pdf, p. 1",
                "type": question_type,
                "difficulty": "Mixed",
            }
            if question_type in {"Multiple choice", "True/False"}:
                question["options"] = ["Correct", "False"] if question_type == "True/False" else ["Correct", "Wrong", "Other", "No"]
            questions.append(question)
        return {"pdf": "ML222.pdf", "topic": "", "questions": questions, "answers": answers}

    @patch("main.grade_quiz")
    @patch("main.get_hybrid_resources")
    def test_all_mcq_payload_is_accepted(self, resources, grade) -> None:
        resources.return_value = ([], object())
        grade.return_value = {"score": 5, "maximum": 5, "results": []}
        response = self.client.post("/api/quiz/grade", json=self._payload(["Correct"] * 5))
        self.assertEqual(response.status_code, 200)

    @patch("main.grade_quiz")
    @patch("main.get_hybrid_resources")
    def test_true_false_short_answer_and_mixed_payloads_are_accepted(self, resources, grade) -> None:
        resources.return_value = ([], object())
        grade.return_value = {"score": 5, "maximum": 5, "results": []}
        for question_type in ("True/False", "Short answer"):
            with self.subTest(question_type=question_type):
                response = self.client.post(
                    "/api/quiz/grade",
                    json=self._payload(["Correct"] * 5, question_type=question_type),
                )
                self.assertEqual(response.status_code, 200)

    def test_malformed_payload_returns_fastapi_validation_error(self) -> None:
        response = self.client.post(
            "/api/quiz/grade",
            json={"pdf": "ML222.pdf", "questions": "not-a-list", "answers": []},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()["detail"], list)

    def test_unanswered_questions_return_readable_error(self) -> None:
        response = self.client.post("/api/quiz/grade", json=self._payload([""] * 5))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Please answer all questions before grading (5 unanswered)")

    def test_mismatched_answers_return_readable_error(self) -> None:
        response = self.client.post("/api/quiz/grade", json=self._payload(["Correct"]))
        self.assertEqual(response.status_code, 422)
        self.assertIn("same length", response.json()["detail"])

    def test_error_helper_handles_structured_errors(self) -> None:
        source = open("static/app.js", encoding="utf-8").read()
        self.assertIn("function formatApiError(error, fallback = \"Request failed\")", source)
        self.assertIn("messages.join(\"; \")", source)
        self.assertIn("typeof error === \"string\"", source)


class QuizWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Document(
                page_content="The model uses self-attention to process tokens.",
                metadata={"filename": "paper.pdf", "page": 2},
            ),
            Document(
                page_content="This enables parallel computation.",
                metadata={"filename": "paper.pdf", "page": 3},
            ),
        ]
        self.questions = [
            {
                "question": f"Question {index}?",
                "answer": f"Reference answer {index}",
                "source": "paper.pdf, p. 3",
            }
            for index in range(1, 6)
        ]
        self.generation_reply = json.dumps(
            {"covered": True, "questions": self.questions}
        )

    def test_generation_is_structured_and_citations_must_be_retrieved(self) -> None:
        prompts: list[str] = []

        def mock_llm(prompt: str) -> str:
            prompts.append(prompt)
            return self.generation_reply

        quiz = generate_quiz("paper.pdf", "attention", self.chunks, ask=mock_llm)

        self.assertEqual(len(prompts), 1)
        self.assertTrue(quiz["covered"])
        self.assertEqual(len(quiz["questions"]), 5)
        self.assertIn("paper.pdf, p. 3", prompts[0])
        self.assertIn("paper.pdf, p. 4", prompts[0])
        self.assertIn("ONLY the retrieved text", prompts[0])
        self.assertEqual(quiz["questions"][0]["source"], "paper.pdf, p. 3")

        invalid_questions = [dict(question) for question in self.questions]
        invalid_questions[0]["source"] = "invented.pdf, p. 99"
        with self.assertRaisesRegex(ValueError, "does not match retrieved context"):
            generate_quiz(
                "paper.pdf",
                "attention",
                self.chunks,
                ask=lambda _: json.dumps(
                    {"covered": True, "questions": invalid_questions}
                ),
            )

    def test_grading_prompt_uses_context_and_not_generated_answer_key(self) -> None:
        context = build_quiz_context(self.chunks)
        prompt = build_grading_prompt(
            self.questions,
            ["My response"] * 5,
            context,
        )

        self.assertIn("ONLY the retrieved source context", prompt)
        self.assertIn("The model uses self-attention", prompt)
        self.assertIn("My response", prompt)
        self.assertNotIn("Reference answer 1", prompt)

    def test_session_state_progresses_from_questions_to_persisted_grade(self) -> None:
        state = make_quiz_state(
            "paper.pdf",
            "attention",
            {"covered": True, "questions": self.questions},
            self.chunks,
        )
        self.assertIsNone(state["grade"])
        self.assertEqual(state["answers"], [""] * 5)
        attempt_id = state["attempt_id"]
        user_answers = ["answer 1", "answer 2", "", "answer 4", "answer 5"]
        replies = [
            {
                "grade": "Correct",
                "feedback": "Correct.",
                "correct_answer": "Answer 1",
                "source": "paper.pdf, p. 3",
            },
            {
                "grade": "Partial",
                "feedback": "Partly right.",
                "correct_answer": "Answer 2",
                "source": "paper.pdf, p. 3",
            },
            {
                "grade": "Incorrect",
                "feedback": "No answer.",
                "correct_answer": "Answer 3",
                "source": "paper.pdf, p. 3",
            },
            {
                "grade": "Correct",
                "feedback": "Correct.",
                "correct_answer": "Answer 4",
                "source": "paper.pdf, p. 3",
            },
            {
                "grade": "Correct",
                "feedback": "Correct.",
                "correct_answer": "Answer 5",
                "source": "paper.pdf, p. 3",
            },
        ]
        calls: list[str] = []

        def mock_grade(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps({"results": replies})

        graded_state = submit_quiz_state(state, user_answers, ask=mock_grade)

        self.assertEqual(len(calls), 1)
        self.assertIn("ONLY the retrieved source context", calls[0])
        self.assertEqual(graded_state["attempt_id"], attempt_id)
        self.assertEqual(graded_state["answers"], user_answers)
        self.assertEqual(graded_state["grade"]["score"], 3.5)
        self.assertEqual(graded_state["grade"]["maximum"], 5)
        self.assertEqual(graded_state["grade"]["results"][2]["grade"], "Incorrect")
        self.assertEqual(
            graded_state["grade"]["results"][2]["source"],
            "paper.pdf, p. 3",
        )

    def test_empty_evidence_skips_generation(self) -> None:
        quiz = generate_quiz(
            "paper.pdf",
            "uncovered topic",
            [],
            ask=lambda _: self.fail("No LLM request should be made without evidence"),
        )
        self.assertEqual(quiz, {"covered": False, "questions": []})

    def test_malformed_json_retries_once_with_valid_count(self) -> None:
        replies = iter(["not json", json.dumps({
            "covered": True,
            "questions": [
                {
                    "question": f"Question {index}?",
                    "answer": f"Answer {index}",
                    "source": "paper.pdf, p. 3",
                    "difficulty": "Medium",
                    "type": "Short answer",
                }
                for index in range(5)
            ],
        })])
        prompts = []

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            return next(replies)

        generated = generate_quiz("paper.pdf", "attention", self.chunks, ask=ask)
        self.assertEqual(len(generated["questions"]), 5)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Generate exactly 5 questions", prompts[0])

    def test_grading_prompt_strictness_follows_difficulty(self) -> None:
        context = build_quiz_context(self.chunks)
        hard_questions = [dict(question, difficulty="Hard") for question in self.questions]
        easy_questions = [dict(question, difficulty="Easy") for question in self.questions]
        hard_prompt = build_grading_prompt(hard_questions, ["answer"] * 5, context)
        easy_prompt = build_grading_prompt(easy_questions, ["answer"] * 5, context)
        self.assertIn("require key terms or correct reasoning", hard_prompt)
        self.assertIn("accept partial paraphrases", easy_prompt)


if __name__ == "__main__":
    unittest.main()
