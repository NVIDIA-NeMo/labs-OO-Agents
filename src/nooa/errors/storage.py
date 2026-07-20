# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialization and storage errors."""

from nooa.errors import NemoOOAgentsError


class SerializationError(NemoOOAgentsError):
    """Error serializing agent state for a snapshot.

    Raised by StorageManager.save_snapshot() when a context-block value
    can't be serialized. Non-serializable agent attributes are not
    fatal — they are logged with a warning and skipped rather than
    raising. To exclude a field from snapshots deliberately, annotate it
    with ``Annotated[T, nosnapshot]`` or set ``__nosnapshot__ = True`` on
    its class (see ``storage/markers.py``).
    """

    pass


class DeserializationError(NemoOOAgentsError):
    """Error deserializing agent state from a snapshot.

    Raised when a serialized value cannot be restored — e.g., the
    envelope references a class not in the allowlist, the class
    cannot be imported, or the data doesn't match the expected format.
    """

    pass


class SnapshotNotFoundError(NemoOOAgentsError):
    """Snapshot ID not found in storage.

    Raised by Agent.load() when StorageManager.restore_snapshot()
    cannot find the given snapshot_id.
    """

    pass


class StorageNotConfiguredError(NemoOOAgentsError):
    """Agent.save() called without a StorageManager.

    Raised when attempting to save an agent that was constructed
    without a ``storage=`` parameter.
    """

    pass
