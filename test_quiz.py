"""Offline tests for grounded quiz generation, submission, and grading."""

import json
import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from quiz import (
    build_grading_prompt,
    build_quiz_context,
    generate_quiz,
    make_quiz_state,
    submit_quiz_state,
)


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

    def test_streamlit_submission_persists_answers_and_renders_grade(self) -> None:
        # Disable dotenv loading so this UI test cannot inspect a developer's
        # local .env file, and replace the grade request with a fixed response.
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        results = [
            {
                "grade": "Correct",
                "feedback": "Supported by the retrieved passage.",
                "correct_answer": f"Reference {index}",
                "source": "gpt4_technical_report.pdf, p. 1",
            }
            for index in range(1, 6)
        ]
        state = {
            "attempt_id": "mocked-ui-attempt",
            "pdf": "gpt4_technical_report.pdf",
            "topic": "",
            "questions": self.questions,
            "answers": [""] * 5,
            "grade": None,
            "context": (
                "Source: gpt4_technical_report.pdf, p. 1\n"
                "The retrieved context supports these answers."
            ),
            "chunks": [
                {
                    "citation": "gpt4_technical_report.pdf, p. 1",
                    "text": "The retrieved context supports these answers.",
                }
            ],
        }
        prompts: list[str] = []

        def mock_call(prompt: str, injected_ask: object, label: str) -> str:
            self.assertIsNone(injected_ask)
            self.assertEqual(label, "quiz_grade")
            prompts.append(prompt)
            return json.dumps({"results": results})

        with patch("quiz._call_llm", side_effect=mock_call):
            app = AppTest.from_file("app.py", default_timeout=30)
            app.session_state["quiz_mode"] = True
            app.session_state["quiz_state"] = state
            app.run()
            self.assertFalse(app.exception, [str(error) for error in app.exception])
            self.assertEqual(len(app.text_area), 5)

            for index, widget in enumerate(app.text_area, start=1):
                widget.set_value(f"User response {index}")
            submit = next(
                button
                for button in app.button
                if button.label == "Submit answers for grading"
            )
            submit.click().run()

        self.assertFalse(app.exception, [str(error) for error in app.exception])
        self.assertEqual(len(prompts), 1)
        self.assertIn("ONLY the retrieved source context", prompts[0])
        self.assertIn("The retrieved context supports these answers.", prompts[0])
        persisted = app.session_state["quiz_state"]
        self.assertEqual(
            persisted["answers"],
            [f"User response {index}" for index in range(1, 6)],
        )
        self.assertEqual(persisted["grade"]["score"], 5.0)
        self.assertTrue(
            any("Final score: 5/5" in item.value for item in app.subheader)
        )


if __name__ == "__main__":
    unittest.main()
