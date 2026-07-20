# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ActorRuntime._generation_lock and QueueManager loop-mismatch fix (gl-212)."""

import asyncio
import threading


class TestGenerationLockLoopChange:
    """Verify ActorRuntime._generation_lock survives event loop changes."""

    def _make_actor(self):
        """Create a minimal ActorRuntime with a fake agent for testing."""
        from nooa.runtime.actor import ActorRuntime

        class FakeAgent:
            pass

        agent = FakeAgent()
        actor = ActorRuntime(agent)
        return actor

    async def test_ensure_noop_when_same_loop(self):
        """_ensure_generation_lock_on_current_loop is a no-op when loop hasn\'t changed."""
        actor = self._make_actor()
        # First call sets the loop
        actor._ensure_generation_lock_on_current_loop()
        original_lock = actor._generation_lock
        original_loop = actor._generation_lock_loop

        # Same loop — lock should NOT be recreated
        actor._ensure_generation_lock_on_current_loop()
        assert actor._generation_lock is original_lock
        assert actor._generation_lock_loop is original_loop

    async def test_ensure_recreates_on_loop_change(self):
        """_ensure_generation_lock_on_current_loop recreates lock when loop differs."""
        actor = self._make_actor()
        # First call sets the loop
        actor._ensure_generation_lock_on_current_loop()
        original_lock = actor._generation_lock

        # Simulate a loop change
        actor._generation_lock_loop = object()  # different object
        actor._ensure_generation_lock_on_current_loop()
        assert actor._generation_lock is not original_lock

    async def test_initial_call_sets_loop(self):
        """First call to _ensure sets _generation_lock_loop from None."""
        actor = self._make_actor()
        assert actor._generation_lock_loop is None
        actor._ensure_generation_lock_on_current_loop()
        assert actor._generation_lock_loop is asyncio.get_running_loop()

    def test_lock_survives_cross_loop_contention(self):
        """Lock contends on loop A, loop closed, contends on loop B."""
        actor = self._make_actor()

        # Phase 1: contend the lock on loop A
        phase1_done = threading.Event()
        phase1_error = [None]

        def loop_A():
            loopA = asyncio.new_event_loop()
            asyncio.set_event_loop(loopA)
            try:

                async def concurrent_acquires():
                    actor._ensure_generation_lock_on_current_loop()

                    async def acquire_and_release():
                        async with actor._generation_lock:
                            await asyncio.sleep(0.01)

                    await asyncio.gather(
                        acquire_and_release(),
                        acquire_and_release(),
                    )

                loopA.run_until_complete(concurrent_acquires())
            except Exception as e:
                phase1_error[0] = e
            finally:
                loopA.close()
                phase1_done.set()

        t1 = threading.Thread(target=loop_A)
        t1.start()
        assert phase1_done.wait(timeout=10), "Phase 1 timed out"
        t1.join(timeout=2)
        assert not t1.is_alive(), "Phase 1 thread still alive after join"
        assert phase1_error[0] is None, f"Phase 1 failed: {phase1_error[0]}"

        # Phase 2: use on loop B (simulates agent loop restart)
        phase2_done = threading.Event()
        phase2_error = [None]

        def loop_B():
            loopB = asyncio.new_event_loop()
            asyncio.set_event_loop(loopB)
            try:

                async def concurrent_acquires():
                    actor._ensure_generation_lock_on_current_loop()

                    async def acquire_and_release():
                        async with actor._generation_lock:
                            await asyncio.sleep(0.01)

                    await asyncio.gather(
                        acquire_and_release(),
                        acquire_and_release(),
                    )

                loopB.run_until_complete(concurrent_acquires())
            except Exception as e:
                phase2_error[0] = e
            finally:
                loopB.close()
                phase2_done.set()

        t2 = threading.Thread(target=loop_B)
        t2.start()
        assert phase2_done.wait(timeout=10), "Phase 2 timed out"
        t2.join(timeout=2)
        assert not t2.is_alive(), "Phase 2 thread still alive after join"
        assert phase2_error[0] is None, f"Cross-loop contention raised: {phase2_error[0]}"


class TestQueueManagerNotifyLoopChange:
    """Verify QueueManager._notify_pair survives event loop changes."""

    async def test_notify_pair_recreated_on_loop_change(self):
        """_notify_pair is recreated when the stored loop differs from current."""
        from nooa.runtime.channels import QueueManager

        qm = QueueManager()
        ch = qm.queue("test_ch")

        # Initialize _notify_pair by setting it manually (simulates first race())
        loop = asyncio.get_running_loop()
        old_event = asyncio.Event()
        qm._notify_pair = (old_event, loop)

        # Simulate loop change by replacing the stored loop with a different ref
        fake_old_loop = object()
        qm._notify_pair = (old_event, fake_old_loop)

        # Put an item so race() returns via fast path (avoids blocking)
        ch.put("item")

        # race() should detect loop mismatch and recreate the pair
        result = await qm.race()
        assert result == [("test_ch", "item")]
        assert qm._notify_pair[1] is loop
        assert qm._notify_pair[0] is not old_event

    async def test_notify_pair_stable_when_same_loop(self):
        """_notify_pair is NOT recreated when loop is the same."""
        from nooa.runtime.channels import QueueManager

        qm = QueueManager()
        ch = qm.queue("test_ch")
        ch.put("first")

        # First race initializes _notify_pair
        await qm.race()
        pair_after_first = qm._notify_pair

        # Second race on same loop should keep the same pair
        ch.put("second")
        await qm.race()
        assert qm._notify_pair is pair_after_first
