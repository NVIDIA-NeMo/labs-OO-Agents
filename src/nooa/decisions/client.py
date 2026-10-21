# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified decision-model contract and HTTP decision client."""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast, runtime_checkable

import httpx
from pydantic import JsonValue

from nooa._immutable_json import freeze
from nooa.decisions.media import DecisionAttachment, decision_media_inputs, validate_attachments
from nooa.decisions.types import Criterion

if TYPE_CHECKING:
    # Importing nooa.unifiedllm loads LiteLLM; keep `import nooa` lightweight.
    from nooa.unifiedllm.retry_config import RetryConfig
    from nooa.unifiedllm.unifiedllm import UnifiedLLM

type DecisionState = str | dict[str, JsonValue] | list[JsonValue]


@dataclass(frozen=True, slots=True)
class BooleanQuestion:
    instructions: Criterion
    criteria: dict[str, Criterion] | None = None
    type: Literal["noul"] = "noul"


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    instructions: Criterion
    criteria: dict[str, Criterion]
    type: Literal["choice"] = "choice"


@dataclass(frozen=True, slots=True)
class ScoreQuestion:
    instructions: Criterion
    criteria: list[Criterion]
    type: Literal["score"] = "score"


type DecisionQuestion = BooleanQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Normalized internal request sent to a decision client."""

    state: DecisionState
    questions: dict[str, DecisionQuestion]
    attachments: tuple[DecisionAttachment, ...] = ()


@dataclass(frozen=True, slots=True)
class BooleanAnswer:
    probability_true: float


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    selected: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    score: float
    probabilities: dict[int, float]
    legend: dict[int, Criterion]
    confidence: float


type DecisionAnswer = BooleanAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    answers: dict[str, DecisionAnswer]
    model: str
    id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # The complete, read-only response body, when the client has one.
    raw: Mapping[str, Any] | None = None


@runtime_checkable
class UnifiedDecisionModel(Protocol):
    """Provider-neutral model interface required by ``DecideStrategy``.

    Decision models expose calibrated answers rather than chat completions.
    Keeping that capability explicit avoids pretending their distributions are
    ordinary generated text.
    """

    # TODO: Share identity, configuration, and lifecycle through a future
    # UnifiedModel abstraction while retaining explicit chat, decision, and
    # vision capabilities on their specialized interfaces.

    model: str

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate a normalized decision request."""
        ...


class DecisionModel(UnifiedDecisionModel):
    """Base class for NOOA decision models.

    Subclasses declare what their answers can support. A model with
    ``provides_probabilities = False`` returns a selected value without
    probability evidence, so ``DecideStrategy`` rejects detailed results and
    thresholds for it before making a request. Clients that implement only
    ``UnifiedDecisionModel`` are treated as providing probabilities.
    """

    #: Whether answers carry real probability evidence.
    provides_probabilities: ClassVar[bool] = True
    #: Recorded as ``DecisionRecord.decision_source`` for this model's calls.
    decision_source: ClassVar[Literal["native", "llm"]] = "native"

    @abstractmethod
    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate a normalized decision request."""

    async def aclose(self) -> None:
        """Release resources owned by this model."""

    @classmethod
    def from_llm(cls, llm: UnifiedLLM | str) -> DecisionModel:
        """Answer decision questions with a chat model.

        The chat model selects one option, boolean, or score level per
        question; it provides no probabilities. Methods returning detailed
        decision objects or using ``Threshold`` therefore raise
        ``DecisionModelRequiredError`` with this model.

        Args:
            llm: A chat client, or a configured model alias.

        Returns:
            A decision model whose calls are recorded with
            ``decision_source="llm"``.
        """
        from nooa.decisions.llm_adapter import LLMDecisionModel

        if isinstance(llm, str):
            from nooa.unifiedllm.registry import get_llm_client

            llm = get_llm_client(llm)
        return LLMDecisionModel(llm)


class DecisionClientError(RuntimeError):
    """Base error raised by decision clients."""


class DecisionTransportError(DecisionClientError):
    """The decision service could not be reached successfully."""


class DecisionAuthenticationError(DecisionClientError):
    """The decision service rejected authentication or authorization."""


class DecisionRefusedError(DecisionClientError):
    """The decision model refused to answer one or more questions.

    Attributes:
        questions: Names of the refused questions.
        raw: The read-only response body, including any answered questions.
    """

    def __init__(self, questions: tuple[str, ...], *, raw: Mapping[str, Any] | None = None) -> None:
        """Record the refused questions and the response they came from."""
        names = ", ".join(repr(name) for name in questions)
        super().__init__(f"The decision model refused to answer {names}")
        self.questions = questions
        self.raw = raw


class InvalidDecisionResponseError(DecisionClientError):
    """The service returned a malformed or incomplete response."""


type DecisionApiStyle = Literal["systemone", "openai"]
type DecisionImageTransport = Literal["json", "typesafe_multipart"]


class DecisionClient(DecisionModel):
    """Async client for HTTP decision-model endpoints.

    ``api_style`` selects the wire format: ``"systemone"`` for System One
    endpoints such as TypeSafe's Jev, or ``"openai"`` for OpenAI's
    ``/v1/decisions``. Both return the same normalized answers.

    The client owns an internally created ``httpx.AsyncClient`` and closes it
    from :meth:`aclose`. A client supplied through ``client=`` remains owned by
    the caller.
    """

    def __init__(
        self,
        model: str,
        *,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        retry_config: RetryConfig | None = None,
        client: httpx.AsyncClient | None = None,
        api_style: DecisionApiStyle = "systemone",
        media_inputs: object = (),
        decision_image_transport: DecisionImageTransport = "json",
        typesafe_danger_zone_images: bool = False,
    ) -> None:
        """Configure a decision client.

        Args:
            model: Decision-model identifier sent with every request.
            endpoint: Complete decisions API URL.
            api_key: Optional bearer token used by an internally created HTTP client.
            timeout: Timeout for an internally created HTTP client, in seconds.
            retry_config: Shared retry policy. Defaults to one transient retry.
            client: Optional caller-owned asynchronous HTTP client.
            api_style: Wire format of ``endpoint``: ``"systemone"`` or ``"openai"``.
            media_inputs: Explicit route-local permission; only ``["image"]``
                is verified for native decisions. Empty means text-only.
            decision_image_transport: ``"json"`` keeps JSON proxies unchanged;
                ``"typesafe_multipart"`` explicitly selects direct TypeSafe's
                local PNG/JPEG binary transport, never inferred by hostname.
            typesafe_danger_zone_images: Explicit experimental TypeSafe image
                opt-in; required along with permission for multipart images.

        Raises:
            ValueError: If ``model`` or ``endpoint`` is empty, or ``api_style``
                is not supported.
        """
        if not model:
            raise ValueError("model must not be empty")
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if api_style not in ("systemone", "openai"):
            raise ValueError(f"api_style must be 'systemone' or 'openai', got {api_style!r}")
        self.media_inputs = decision_media_inputs(media_inputs)
        if decision_image_transport not in ("json", "typesafe_multipart"):
            raise ValueError("decision_image_transport must be json or typesafe_multipart")
        if decision_image_transport == "typesafe_multipart" and api_style != "systemone":
            raise ValueError("typesafe_multipart requires api_style=systemone")
        if not isinstance(typesafe_danger_zone_images, bool):
            raise ValueError("typesafe_danger_zone_images must be an explicit boolean")
        if typesafe_danger_zone_images and decision_image_transport != "typesafe_multipart":
            raise ValueError("typesafe_danger_zone_images requires typesafe_multipart transport")
        self.decision_image_transport = decision_image_transport
        self.typesafe_danger_zone_images = typesafe_danger_zone_images
        self.api_style = api_style
        self.model = model
        self.endpoint = endpoint
        from nooa.unifiedllm.retry_config import RetryConfig

        self.retry_config = retry_config or RetryConfig(
            max_retries=1,
            base_delay=0.1,
            jitter_factor=0.0,
            rate_limit_extra_retries=0,
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            retryable_exceptions=(httpx.TransportError,),
        )
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=headers)

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate a normalized decision request and validate the response.

        Raises:
            DecisionAuthenticationError: If the endpoint returns 401 or 403.
            DecisionTransportError: If transport or HTTP handling fails.
            DecisionRefusedError: If an OpenAI-style endpoint refuses a question.
            InvalidDecisionResponseError: If the response body is malformed.
        """
        decoded = validate_attachments(request.attachments, self.media_inputs)
        # Validate state before retry/HTTP even for direct or middleware requests.
        from nooa.decisions.schema import _json_value

        _json_value(request.state)
        if request.attachments:
            if not isinstance(request.state, dict) or not isinstance(
                request.state.get("inputs"), dict
            ):
                raise ValueError(
                    "Decision image attachments require structured state.inputs named references"
                )
            inputs = request.state["inputs"]
            associations = {
                name: attachment.alias
                for attachment in request.attachments
                for name in attachment.input_names
            }
            aliases = {attachment.alias for attachment in request.attachments}
            if any(inputs.get(name) != alias for name, alias in associations.items()) or any(
                isinstance(value, str) and value in aliases and associations.get(name) != value
                for name, value in inputs.items()
            ):
                raise ValueError(
                    "Decision image input names must be referenced by their exact attachment aliases in state.inputs"
                )
            if self.api_style == "systemone":
                if self.decision_image_transport != "typesafe_multipart":
                    raise ValueError(
                        "System One JSON blocks image requests before sending. Use typesafe_multipart for an approved direct TypeSafe connection, or leave images off."
                    )
                if not self.typesafe_danger_zone_images:
                    raise ValueError(
                        "TypeSafe images require explicit typesafe_danger_zone_images=True opt-in"
                    )
        if self.api_style == "openai":
            from nooa.decisions._openai_wire import openai_payload

            payload = openai_payload(self.model, request)
        else:
            payload = {
                "model": self.model,
                "state": request.state,
                "questions": {
                    name: self._question_payload(question)
                    for name, question in request.questions.items()
                },
            }
        files = None
        if request.attachments and self.decision_image_transport == "typesafe_multipart":
            assets: dict[str, Any] = {}
            image_parts: list[tuple[str, Any]] = []
            for index, (attachment, data) in enumerate(
                zip(request.attachments, decoded, strict=True)
            ):
                if data is None:
                    raise ValueError(
                        "TypeSafe multipart requires locally encoded data URL bytes; remote image URLs are not fetched"
                    )
                signature = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}.get(
                    attachment.media_type
                )
                if signature is None or not data.startswith(signature):
                    raise ValueError(
                        "TypeSafe multipart images must have matching PNG/JPEG MIME and file signature"
                    )
                if attachment.detail is not None:
                    raise ValueError(
                        "TypeSafe multipart cannot send an image detail setting. Leave detail unset."
                    )
                name = f"asset-{index}"
                assets[attachment.alias] = {"type": "image", "part": name}
                image_parts.append((name, (name, data, attachment.media_type)))
            payload["assets"] = assets
            files = [
                (
                    "request",
                    (None, json.dumps(payload, ensure_ascii=False).encode(), "application/json"),
                ),
                *image_parts,
            ]
        try:
            from nooa.unifiedllm.retry import with_retry

            if files is None:
                response = await with_retry(self._post, payload, config=self.retry_config)
            else:
                response = await with_retry(self._post_multipart, files, config=self.retry_config)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                raise DecisionAuthenticationError(
                    f"Decision service returned HTTP {exc.response.status_code}"
                ) from exc
            raise DecisionTransportError(
                f"Decision service returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.TransportError as exc:
            raise DecisionTransportError(f"Decision request failed: {exc}") from exc

        try:
            if self.api_style == "openai":
                from nooa.decisions._openai_wire import parse_openai_response

                return parse_openai_response(
                    response.json(), request, request_id=response.headers.get("x-request-id")
                )
            return self._parse_response(response.json(), request)
        except (AttributeError, ValueError, TypeError, KeyError) as exc:
            raise InvalidDecisionResponseError(f"Invalid decision response: {exc}") from exc

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """Send one HTTP attempt and raise for non-success status codes."""
        response = await self._client.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response

    async def _post_multipart(self, files: list[tuple[str, Any]]) -> httpx.Response:
        """Send repeatable byte parts; httpx owns boundary and Content-Type."""
        # Generate the multipart headers independently of caller defaults (a
        # caller-owned JSON proxy client may have Content-Type: application/json).
        multipart = httpx.Request("POST", self.endpoint, files=files)
        content = multipart.read()
        response = await self._client.post(
            self.endpoint,
            content=content,
            headers={
                "Content-Type": multipart.headers["Content-Type"],
                "Content-Length": str(len(content)),
                "TypeSafe-Danger-Zone-Images": "true",
            },
        )
        response.raise_for_status()
        return response

    async def aclose(self) -> None:
        """Close the internally owned HTTP client, if any."""
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _question_payload(question: DecisionQuestion) -> dict[str, Any]:
        """Convert one normalized question into the decision wire format."""
        payload: dict[str, Any] = {
            "type": question.type,
            "instructions": question.instructions,
        }
        if question.criteria is not None:
            payload["criteria"] = question.criteria
        return payload

    @staticmethod
    def _parse_response(data: Any, request: DecisionRequest) -> DecisionResponse:
        """Normalize and validate a response against the requested questions."""
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise TypeError("expected an object containing an answers object")
        raw_answers = data["answers"]
        missing = set(request.questions) - set(raw_answers)
        extra = set(raw_answers) - set(request.questions)
        if missing or extra:
            raise ValueError(f"answer names differ from request (missing={missing}, extra={extra})")

        answers: dict[str, DecisionAnswer] = {}
        for name, question in request.questions.items():
            answer = raw_answers[name]
            if not isinstance(answer, dict):
                raise TypeError(f"answer {name!r} must be an object")
            if isinstance(question, BooleanQuestion):
                answers[name] = BooleanAnswer(probability_true=float(answer["noul"]))
            elif isinstance(question, ChoiceQuestion):
                answers[name] = ChoiceAnswer(
                    selected=str(answer["choice"]),
                    probabilities={str(k): float(v) for k, v in answer["probabilities"].items()},
                    confidence=float(answer["confidence"]),
                )
            else:
                answers[name] = ScoreAnswer(
                    score=float(answer["score"]),
                    probabilities={int(k): float(v) for k, v in answer["probabilities"].items()},
                    legend={int(k): v for k, v in answer["legend"].items()},
                    confidence=float(answer["confidence"]),
                )

        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            raise TypeError("usage must be an object")
        return DecisionResponse(
            answers=answers,
            model=str(data.get("model") or ""),
            id=str(data["id"]) if data.get("id") is not None else None,
            usage=usage,
            raw=cast("Mapping[str, Any]", freeze(data)),
        )


__all__ = [
    "DecisionAttachment",
    "DecisionAuthenticationError",
    "DecisionClient",
    "DecisionClientError",
    "DecisionModel",
    "DecisionRefusedError",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionTransportError",
    "InvalidDecisionResponseError",
    "UnifiedDecisionModel",
]
