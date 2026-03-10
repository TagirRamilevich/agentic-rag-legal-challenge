import os

FREE_TEXT_MAX_LEN = 280


def validate_submission(
    submission: dict,
    docs_dir: str = None,
    page_counts: dict = None,
) -> list[str]:
    errors = []

    if "version" in submission:
        errors.append("submission MUST NOT contain 'version' field")

    arch = submission.get("architecture_summary", "")
    if arch and len(arch) > 500:
        errors.append(f"architecture_summary exceeds 500 chars ({len(arch)})")

    if "answers" not in submission:
        errors.append("missing 'answers' field")
        return errors

    for i, ans in enumerate(submission["answers"]):
        p = f"answers[{i}]"

        if "question_id" not in ans:
            errors.append(f"{p}: missing question_id")
        if "answer" not in ans:
            errors.append(f"{p}: missing answer field")

        tel = ans.get("telemetry")
        if tel is None:
            errors.append(f"{p}: missing telemetry")
            continue

        timing = tel.get("timing")
        if timing is None:
            errors.append(f"{p}.telemetry: missing timing")
        else:
            for field in ("ttft_ms", "tpot_ms", "total_time_ms"):
                if field not in timing:
                    errors.append(f"{p}.telemetry.timing: missing {field}")
                elif not isinstance(timing[field], (int, float)):
                    errors.append(f"{p}.telemetry.timing.{field} must be numeric")

        retrieval = tel.get("retrieval")
        if retrieval is None:
            errors.append(f"{p}.telemetry: missing retrieval")
        else:
            refs = retrieval.get("retrieved_chunk_pages")
            if refs is None:
                errors.append(f"{p}.telemetry.retrieval: missing retrieved_chunk_pages")
            else:
                for ref in refs:
                    doc_id = ref.get("doc_id", "")
                    page_numbers = ref.get("page_numbers", [])
                    if not doc_id:
                        errors.append(f"{p}: retrieval ref missing doc_id")
                    if not isinstance(page_numbers, list):
                        errors.append(f"{p}: page_numbers must be a list")
                    if docs_dir and doc_id:
                        if not os.path.exists(os.path.join(docs_dir, doc_id)):
                            errors.append(f"{p}: doc_id '{doc_id}' not found in {docs_dir}")
                    if page_counts and doc_id and isinstance(page_numbers, list):
                        max_p = page_counts.get(doc_id, 0)
                        for pn in page_numbers:
                            if not isinstance(pn, int) or pn < 1 or pn > max_p:
                                errors.append(
                                    f"{p}: page {pn} out of range for {doc_id} (max {max_p})"
                                )

        usage = tel.get("usage")
        if usage is None:
            errors.append(f"{p}.telemetry: missing usage")
        else:
            for field in ("input_tokens", "output_tokens"):
                if field not in usage:
                    errors.append(f"{p}.telemetry.usage: missing {field}")

        answer_val = ans.get("answer")
        if isinstance(answer_val, str) and len(answer_val) > FREE_TEXT_MAX_LEN:
            errors.append(
                f"{p}: answer string length {len(answer_val)} exceeds {FREE_TEXT_MAX_LEN}"
            )

    return errors
