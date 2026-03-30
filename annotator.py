"""DSPy signature for LLM-based privacy annotation (teacher model)."""

from typing import Literal

import dspy


class Annotate(dspy.Signature):
    """Provide a 'privacy rating' scoring on how sensitive or private the
    user_text, for example including personal information or indirectly identifiable
    information.
    Give your answer on a scale of 1 to 5, where 1 means that the user_text is not
    sensitive or private at all, and 5 means that text contains highly sensitive or private
    information.
    Here is the scale you should use to build your answer:
    1: The user_text is harmless: completely free of any private or sensitive information,
    either direct or indirect identifiers.
    2: The user_text is mostly not private: may contain some indirect identifiers, but
    is mostly free of sensitive information.
    3: The user_text is somewhat private: contains some direct or indirect identifiers,
    and can be deemed somewhat personal information.
    4: The user_text is very private: contains several direct or indirect identifiers, and
    it is clear that this text contains personal information.
    5: The user_text is extremely private: contains highly sensitive information, such
    as direct personal identifiers, and the text is clearly something that should not be
    made public."""

    user_text: str = dspy.InputField()
    privacy_rating: Literal["1", "2", "3", "4", "5"] = dspy.OutputField()


annotator_predict = dspy.Predict(Annotate)
